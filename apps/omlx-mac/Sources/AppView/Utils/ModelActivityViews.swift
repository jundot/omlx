import SwiftUI

struct ModelActivityBadge: View {
    let snapshot: ModelActivitySnapshot
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        StatusPill(status: .custom(color: tint, label: snapshot.badge, fillBg: true))
    }

    private var tint: Color {
        switch snapshot.badgePhase {
        case .prefill:              return theme.blueDot
        case .generating, .working: return theme.greenDot
        case .queued:               return theme.amberDot
        }
    }
}

struct ModelActivityList: View {
    let snapshot: ModelActivitySnapshot
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            ForEach(snapshot.requests) { ModelActivityRow(request: $0) }
            if snapshot.queued > 0 {
                HStack(spacing: 8) {
                    Circle().fill(theme.amberDot).frame(width: 5, height: 5)
                    Text(ActivityFormat.queuedDetail(count: snapshot.queued))
                        .font(.omlxMono(10.5))
                        .foregroundStyle(theme.textTertiary)
                    Spacer(minLength: 0)
                }
            }
        }
        // Keyed on the row set, not the snapshot: the numbers change every
        // poll and cross-fading those would smear the text.
        .animation(.easeInOut(duration: 0.18), value: rowIdentity)
    }

    private var rowIdentity: [String] {
        snapshot.requests.map(\.id) + ["queued:\(snapshot.queued)"]
    }
}

private struct ModelActivityRow: View {
    let request: ModelRequestActivity
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 8) {
            Text(request.shortID)
                .font(.omlxMono(10.5))
                .foregroundStyle(theme.textTertiary)
                .frame(width: 60, alignment: .leading)

            if let fraction = request.fraction, request.isLive {
                HStack(spacing: 6) {
                    Text(ActivityFormat.badge(for: .prefill))
                        .font(.omlxMono(10.5))
                        .foregroundStyle(tint)
                        .lineLimit(1)
                    ProgressBar(progress: fraction, tint: tint)
                        .frame(width: 70)
                    Text(request.percentText ?? "")
                        .font(.omlxMono(10.5))
                        .foregroundStyle(tint)
                        .frame(width: 32, alignment: .trailing)
                }
            } else {
                HStack(spacing: 6) {
                    Circle().fill(tint).frame(width: 5, height: 5)
                    Text(label).font(.omlxMono(10.5)).lineLimit(1)
                }
                .foregroundStyle(tint)
                // 70 + 8 + 32, so the detail column lines up with prefill rows.
                .frame(width: 110, alignment: .leading)
            }

            Text(request.detail)
                .font(.omlxMono(10.5))
                .foregroundStyle(theme.textSecondary)
                .lineLimit(1)
            Spacer(minLength: 0)
        }
        .opacity(request.isLive ? 1 : 0.55)
    }

    private var label: String {
        switch request.phase {
        case .prefill:    return ActivityFormat.badge(for: .prefill)
        case .generating: return ActivityFormat.badge(for: .generating)
        case .working:    return ActivityFormat.badge(for: .working)
        case .finished:   return ActivityFormat.doneBadge
        }
    }

    private var tint: Color {
        guard request.isLive else { return theme.textTertiary }
        switch request.phase {
        case .prefill:  return theme.blueDot
        default:        return theme.greenDot
        }
    }
}
