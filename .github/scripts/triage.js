// Issue and PR triage rules. See docs/CONTRIBUTING.md and discussion #4256.

const POLICY = '#4256';
const CONTRIBUTING_URL = 'https://github.com/jundot/omlx/blob/main/docs/CONTRIBUTING.md';
const IDEAS_URL = 'https://github.com/jundot/omlx/discussions/categories/ideas';

const TRUSTED_ASSOCIATIONS = new Set(['OWNER', 'MEMBER', 'COLLABORATOR', 'CONTRIBUTOR']);
const LARGE_PR_LINES = 1000;
// The large PR rule applies to PRs opened after the policy announcement.
const LARGE_PR_SINCE = Date.parse('2026-10-05T00:00:00Z');
const LARGE_PR_MARKER = '<!-- omlx-triage:large-pr -->';
const DRAFT_IDLE_DAYS = 30;
const CONFLICT_IDLE_DAYS = 30;
const PR_IDLE_DAYS = 60;
const ISSUE_EXEMPT_LABELS = ['bug', 'planned'];
const PR_EXEMPT_LABELS = ['planned'];
// Stay under the secondary limit of about 500 content-creating requests per hour.
const WRITE_INTERVAL_MS = 8000;
// Guard against a bad cutoff closing most of the tracker.
const MAX_OUTDATED_SHARE = 0.7;
const DAY_MS = 24 * 60 * 60 * 1000;

const LARGE_PR_MESSAGE =
  'This PR changes over 1,000 lines, and large PRs from first-time contributors need an issue or discussion first ' +
  `so the approach can be checked before review ([CONTRIBUTING.md](${CONTRIBUTING_URL})). ` +
  'It has been moved to draft for now. ' +
  `Please open an issue or an [Ideas](${IDEAS_URL}) discussion and link it here. ` +
  'Once the approach is agreed, mark it ready for review.\n\n' +
  LARGE_PR_MARKER;

const PR_CLOSE_REASONS = {
  conflict: `Closing this PR because it has merge conflicts and no new commits for over ${CONFLICT_IDLE_DAYS} days.`,
  idle: `Closing this PR because it has had no new commits for over ${PR_IDLE_DAYS} days.`,
  draft: `Closing this draft PR because it has had no new commits for over ${DRAFT_IDLE_DAYS} days.`,
};
const PR_CLOSE_TAIL =
  "This isn't a judgment on the change. If you'd like to continue, rebase on current `main` and open a new PR. " +
  `See ${POLICY} for the policy.`;

const OPEN_PRS_QUERY = `
query($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequests(states: OPEN, first: 50, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id number isDraft mergeable authorAssociation createdAt additions deletions
        author { __typename login }
        labels(first: 20) { nodes { name } }
        commits(last: 1) { nodes { commit { committedDate } } }
        closingIssuesReferences(first: 20) { nodes { number state labels(first: 20) { nodes { name } } } }
      }
    }
  }
}`;

const OPEN_ISSUES_QUERY = `
query($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    issues(states: OPEN, first: 50, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number createdAt
        labels(first: 20) { nodes { name } }
        comments(last: 10) { nodes { createdAt author { __typename } } }
      }
    }
  }
}`;

const MERGEABLE_QUERY = `
query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) { pullRequest(number: $number) { mergeable } }
}`;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const labelNames = (node) => node.labels.nodes.map((l) => l.name);

module.exports = async ({ github, context, core }) => {
  const { owner, repo } = context.repo;
  const live = process.env.TRIAGE_LIVE === 'true';
  const counts = {};
  const count = (key) => (counts[key] = (counts[key] || 0) + 1);

  async function fetchAll(query, field) {
    const nodes = [];
    let cursor = null;
    for (;;) {
      const data = await github.graphql(query, { owner, repo, cursor });
      const page = data.repository[field];
      nodes.push(...page.nodes);
      if (!page.pageInfo.hasNextPage) return nodes;
      cursor = page.pageInfo.endCursor;
    }
  }

  const comment = (number, body) => github.rest.issues.createComment({ owner, repo, issue_number: number, body });
  const addLabel = (number, name) => github.rest.issues.addLabels({ owner, repo, issue_number: number, labels: [name] });
  async function removeLabel(number, name) {
    try {
      await github.rest.issues.removeLabel({ owner, repo, issue_number: number, name });
    } catch (err) {
      if (err.status !== 404) throw err;
    }
  }

  // Scheduled writes run only when TRIAGE_LIVE is true, spaced out with retries on rate limits.
  async function scheduledWrite(key, description, fn) {
    count(key);
    if (!live) {
      core.info(`[dry-run] ${description}`);
      return;
    }
    core.info(description);
    for (let attempt = 1; ; attempt++) {
      try {
        await fn();
        break;
      } catch (err) {
        if (attempt > 3 || (err.status !== 403 && err.status !== 429)) throw err;
        core.warning(`${description}: HTTP ${err.status}, retrying in ${attempt} min`);
        await sleep(attempt * 60 * 1000);
      }
    }
    await sleep(WRITE_INTERVAL_MS);
  }

  // Each PR is moved to draft at most once; the marker comment records that it was asked.
  async function draftLargePr(pr) {
    if (pr.isDraft || pr.author?.__typename === 'Bot' || TRUSTED_ASSOCIATIONS.has(pr.authorAssociation)) return;
    if (Date.parse(pr.createdAt) < LARGE_PR_SINCE || pr.additions + pr.deletions <= LARGE_PR_LINES) return;
    const comments = await github.paginate(github.rest.issues.listComments, { owner, repo, issue_number: pr.number, per_page: 100 });
    if (comments.some((c) => c.body?.includes(LARGE_PR_MARKER))) return;
    const lines = pr.additions + pr.deletions;
    await scheduledWrite('draft large PR', `move PR #${pr.number} to draft (${lines} lines, ${pr.authorAssociation})`, async () => {
      await github.graphql(
        'mutation($id: ID!) { convertPullRequestToDraft(input: { pullRequestId: $id }) { pullRequest { isDraft } } }',
        { id: pr.id },
      );
      await comment(pr.number, LARGE_PR_MESSAGE);
    });
  }

  async function onIssueComment() {
    const { issue, comment: reply } = context.payload;
    if (issue.pull_request || reply.user.login !== issue.user.login) return;
    if (issue.labels.some((l) => l.name === 'needs-info')) await removeLabel(issue.number, 'needs-info');
  }

  const prExempt = (pr) =>
    pr.author?.__typename === 'Bot' ||
    pr.authorAssociation === 'OWNER' ||
    labelNames(pr).some((l) => PR_EXEMPT_LABELS.includes(l));

  function prIdleDays(pr, now) {
    const last = pr.commits.nodes[0]?.commit.committedDate;
    return last ? (now - Date.parse(last)) / DAY_MS : 0;
  }

  function prCloseReason(pr, now) {
    if (prExempt(pr)) return null;
    const idleDays = prIdleDays(pr, now);
    if (pr.isDraft && idleDays >= DRAFT_IDLE_DAYS) return 'draft';
    if (pr.mergeable === 'CONFLICTING' && idleDays >= CONFLICT_IDLE_DAYS) return 'conflict';
    if (idleDays >= PR_IDLE_DAYS) return 'idle';
    return null;
  }

  // GitHub computes mergeability lazily after main moves and reports UNKNOWN until it is done.
  async function resolveMergeable(prs, now) {
    for (let round = 0; round < 3; round++) {
      const pending = prs.filter((pr) => {
        const idleDays = prIdleDays(pr, now);
        return pr.mergeable === 'UNKNOWN' && !pr.isDraft && !prExempt(pr) &&
          idleDays >= CONFLICT_IDLE_DAYS && idleDays < PR_IDLE_DAYS;
      });
      if (pending.length === 0) return;
      core.info(`Waiting for mergeability of ${pending.length} PRs`);
      await sleep(30 * 1000);
      for (const pr of pending) {
        const data = await github.graphql(MERGEABLE_QUERY, { owner, repo, number: pr.number });
        pr.mergeable = data.repository.pullRequest.mergeable;
      }
    }
  }

  // The first stable release of the line before the newest one. Tags like v0.7.0rc1 are not stable.
  function previousLineStart(releases) {
    const lines = new Map();
    for (const r of releases) {
      const m = /^v(\d+)\.(\d+)\.(\d+)$/.exec(r.tag_name);
      if (!m || r.draft || !r.published_at) continue;
      const key = `${m[1]}.${m[2]}`;
      const at = Date.parse(r.published_at);
      const known = lines.get(key);
      if (!known || at < known.at) lines.set(key, { major: +m[1], minor: +m[2], version: r.tag_name.slice(1), at });
    }
    const sorted = [...lines.values()].sort((a, b) => a.major - b.major || a.minor - b.minor);
    return sorted.length >= 2 ? sorted[sorted.length - 2] : null;
  }

  // Creation or the latest comment by a person. Bot comments, such as stale notices, do not count.
  function lastActivity(issue) {
    const times = issue.comments.nodes
      .filter((c) => c.author?.__typename !== 'Bot')
      .map((c) => Date.parse(c.createdAt));
    return Math.max(Date.parse(issue.createdAt), ...times);
  }

  async function daily() {
    const now = Date.now();

    const prs = await fetchAll(OPEN_PRS_QUERY, 'pullRequests');
    await resolveMergeable(prs, now);

    const openPrs = [];
    for (const pr of prs) {
      const reason = prCloseReason(pr, now);
      if (!reason) {
        openPrs.push(pr);
        continue;
      }
      await scheduledWrite(`close PR (${reason})`, `close PR #${pr.number} (${reason})`, async () => {
        await comment(pr.number, `${PR_CLOSE_REASONS[reason]} ${PR_CLOSE_TAIL}`);
        await github.rest.pulls.update({ owner, repo, pull_number: pr.number, state: 'closed' });
      });
    }

    for (const pr of openPrs) await draftLargePr(pr);

    const linked = new Map();
    for (const pr of openPrs) {
      for (const issue of pr.closingIssuesReferences.nodes) {
        if (issue.state === 'OPEN') linked.set(issue.number, labelNames(issue));
      }
    }
    for (const [number, labels] of linked) {
      if (!labels.includes('has-pr')) {
        await scheduledWrite('add has-pr', `add has-pr to #${number}`, () => addLabel(number, 'has-pr'));
      }
    }

    const issues = await fetchAll(OPEN_ISSUES_QUERY, 'issues');
    for (const issue of issues) {
      if (labelNames(issue).includes('has-pr') && !linked.has(issue.number)) {
        await scheduledWrite('remove has-pr', `remove has-pr from #${issue.number}`, () =>
          removeLabel(issue.number, 'has-pr'),
        );
      }
    }

    const releases = await github.paginate(github.rest.repos.listReleases, { owner, repo, per_page: 100 });
    const line = previousLineStart(releases);
    if (line) {
      const released = new Date(line.at).toLocaleDateString('en-US', { month: 'long', day: 'numeric', timeZone: 'UTC' });
      const message =
        `Closing this because its last activity was before ${line.version} (released ${released}), ` +
        "and a lot has changed since then. If it still happens on the latest release, or you think it needs another look, " +
        `please open a new issue and link this one. See ${POLICY} for the policy.`;
      const outdated = issues.filter(
        (i) =>
          !linked.has(i.number) &&
          !labelNames(i).some((l) => ISSUE_EXEMPT_LABELS.includes(l)) &&
          lastActivity(i) < line.at,
      );
      core.info(`Version rule: last activity before ${line.version} (${released}), ${outdated.length} of ${issues.length} open issues`);
      if (outdated.length > MAX_OUTDATED_SHARE * issues.length) {
        throw new Error(`Version rule matched ${outdated.length} of ${issues.length} open issues; refusing to close them`);
      }
      for (const issue of outdated) {
        await scheduledWrite('close outdated issue', `close #${issue.number} as outdated`, async () => {
          await comment(issue.number, message);
          await addLabel(issue.number, 'outdated');
          await github.rest.issues.update({ owner, repo, issue_number: issue.number, state: 'closed', state_reason: 'not_planned' });
        });
      }
    }

    const rows = Object.entries(counts).map(([k, v]) => [k, String(v)]);
    await core.summary
      .addHeading(live ? 'Triage' : 'Triage (dry-run)')
      .addTable([[{ data: 'Action', header: true }, { data: 'Count', header: true }], ...rows])
      .write();
  }

  if (context.eventName === 'issue_comment') return onIssueComment();
  return daily();
};
