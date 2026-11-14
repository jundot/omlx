# Web UI design rules

These rules cover the web UI in `apps/omlx-web/omlx_web`. Colors and type steps apply to every page, including the chat page and the top navbar. The button, choice control, badge and corner rules cover login and the dashboard; the chat page and the navbar keep their own shapes. Use them when you add or change a template, and rebuild with `make web`; a CSS-only change can use `cd apps/omlx-web && python build_css.py`.

## Colors

The palette and the light semantic colors are defined in `apps/omlx-web/omlx_web/static/css/theme.css`, which `base.html` loads before `tailwind.css`. `dashboard.css` adds the component classes and the dark values of the neutral utility channels; only `dashboard.html` loads it, so until those values move into `theme.css` (#4400), the chat page uses the semantic utilities below and login keeps a hand-copied dark block.

- Palette entries are named `--palette-<family>-<step>` and hold `R G B` channels, so both `rgb(var(--palette-blue-500))` and `rgb(var(--palette-blue-500) / 0.2)` work.
- Semantic tokens name a role and point at the palette: `--bg-primary`, `--bg-secondary`, `--bg-tertiary`, `--text-primary` ... `--text-muted`, `--border-faint`, `--border-normal`, `--code-bg`, `--link-color`, `--btn-primary*`, `--switch-on`, `--text-danger`, `--bg-danger-hover`, `--timeline-accent`, `--focus-ring-color`, and `--text-success` (an alias of `--success-text`). `--bg-danger-hover` is a role token but is not editable in Theme Settings yet.
- The status roles are `--success`, `--warning`, `--danger` and `--info`, each with `-soft`, `-line` and `-text` values. `status.css` maps the green, amber, red and blue utilities to a per-step channel (`--<role>-bg-<step>`, `--<role>-fg-<step>`, `--<role>-bd-<step>`); a step without a channel keeps the palette value. When a role is overridden, `custom_theme.js` recomputes the ladder and derives the badge shades and the memory-meter stops.
- `[data-theme="dark"]` in the same file gives the dark values of the semantic tokens and switches `color-scheme`. The neutral channels' dark values live in `dashboard.css` and therefore apply to the dashboard only; moving them into `theme.css` is tracked in #4400. The chat page uses the semantic utilities `bg-surface`, `bg-surface-alt`, `bg-surface-muted`, `text-fg`, `text-fg-secondary`, `text-fg-tertiary`, `text-fg-muted`, `border-line`, `border-line-strong`, `bg-accent`, `text-accent-fg` and `bg-danger-bg`; `bg-code` is used in the dashboard settings.
- Only these neutral utilities resolve through the `*-scale-*` channels: `bg-white`, `bg-neutral-50/100/200`, `border-neutral-100/200/300` (and the matching `divide-*`), the `from-/to-neutral-50/100` gradient stops, and `text-neutral-300` through `text-neutral-900`. `bg-neutral-300` and above are fixed palette values. `tailwind.config.js` maps the 16 Tailwind hue families plus `white` and `black`; `slate`, `zinc`, `stone`, `yellow`, `lime` and `fuchsia` are not mapped and keep Tailwind's defaults.
- Families outside Tailwind's palette: `night-*` (dark mode surfaces and text), `signal-red-*` (the readability red), `mid-gray` (empty cells in the usage strip), and the non-standard `zinc-450` step used by the dark channels.

Where to use what:

| Place | Use |
|---|---|
| Dashboard and login templates | Tailwind color utilities (`text-neutral-500`, `bg-amber-50`) |
| Chat templates | Semantic utilities (`bg-surface`, `text-fg-secondary`, `border-line`, `bg-accent`) |
| Stylesheets | Semantic tokens, or palette entries for one-off marks |
| Styles built in JS | `rgb(var(--palette-x) / alpha)` |

Do not write hex or `rgb()` literals outside `theme.css`. The exceptions: the Enhanced Readability block in `base.html`, whose selectors match inline style text; the native select chevron data URI in `theme.css`, which cannot read a variable and must be edited in both theme blocks; the switch knob in the Preview canvas (`#ffffff`, white by design); the `#000000` token-resolution fallback; the example JSON in the Theme Settings placeholder; and the vendored hljs stylesheets, whose syntax colors do not follow the custom theme.

### Danger text

A warning block uses the first two steps below, so it reads as one level of emphasis instead of three shades of red:

| Step | Use |
|---|---|
| `text-red-700` | The label of a setting that is dangerous to turn on, such as Settings → Auth & Info → Allow inference without API key |
| `text-red-600` | Its supporting text, and inline error strings. 4.83:1 on white, so it clears the 4.5:1 text floor at 12px |
| `text-red-500` | Accents only: hover, borders, rings, icons. As text at 12–14px it is 3.76:1 on white, under the 4.5:1 floor |

These steps are the danger role: `status.css` resolves `text-red-500/600/700` through `--danger-fg-*`, so a custom danger color moves them together.

## Custom theme

`ui.custom_theme` stores per-theme token overrides (`{"light": {"bg-primary": "#..."}, "dark": {...}}`), and `static/js/custom_theme.js` paints them before first paint. Two lists own the contract: `GROUPS` holds the 22 tokens the editor exposes, and `DERIVED` holds the 12 badge shades computed from the four status roles; `CHANNELS` binds the neutral utility channels to their token. `--text-success` follows `--success-text` and is not edited directly.

`CONTRAST_RULES` uses 4.5:1 for `text-primary`, `text-secondary`, `link-color`, `btn-primary-text` and the four `<role>-text` values; 3:1 for `text-tertiary`, `text-danger`, `focus-ring-color`, `switch-on` and `timeline-accent`; muted text and borders have no floor, because the stock theme draws them below 3:1. The editor disables Apply while a rule fails, names the failing pairs and outlines the fields to change.

The editor shows one column per mode. Column controls read the inline `--theme-cell-*` values (background, text, border, placeholder, surface, line, icon, icon-hover, hover, ring, danger) and the headers use the stock palette, so editing a surface token does not flip the column. Invalid hex gets a red border. Loaded JSON is merged into the draft token by token; restoring the defaults asks for confirmation. Clicking a part of the Preview focuses its input, so every part must be registered in `theme_preview.js` `hit()`.

The server checks only the shape: `#rrggbb` (six digits), a token name matching `[a-z][a-z0-9-]{0,47}`, and at most 64 entries per mode. Contrast is the editor's job.

### How the dark draft is generated

**Generate dark from light** fills the dark column from the light one in OKLCH (`darkFromLight` in `custom_theme.js`). It only edits the draft, so Apply still runs the contrast check. The constants are calibrated so the stock light palette lands close to the stock dark one.

1. Neutrals invert their lightness (`1 - L`) through `NEUTRAL_CURVE`, keep their hue and cap chroma at 0.012. The switch ON track stops at L 0.72 because its knob stays white.
2. Status fills keep their color unless darker than L 0.62; links, the accent and danger text are lifted to at least L 0.71.
3. A color outside sRGB loses chroma until it fits.
4. A foreground below its contrast floor steps its lightness away from its backgrounds until it passes.

The Preview header's Save as image button writes both modes, every editable token's English name and exact hex, and the same token→hex map as JSON in an `omlx-theme` `tEXt` chunk. The table lists the 22 `GROUPS` tokens; the JSON contains only those tokens, not the derived values or the labels. The image is English-only so it can be handed to an agent or shared outside the dashboard. `theme_preview.js` draws the preview card for both: the Preview canvases in the UI language, the image with its fixed English labels, so the two never drift apart.

### Adding a token

1. `theme.css`: light in `:root`, dark in `[data-theme="dark"]`.
2. `tailwind.config.js` if it needs a utility; rebuild with `make web`.
3. `GROUPS`, or `DERIVED` / `CHANNELS` when it is derived or drives a neutral channel.
4. `CONTRAST_RULES` when it carries text or controls; `ACCENT_MIN_L` when it is an accent.
5. `i18n/en.json`: `settings.theme.token.<id>` and, for a new group, `settings.theme.group.<id>`; the other locales follow.
6. `theme_preview.js`: `TOKEN_LABELS`, the drawing and its `hit()` box.

## Type

| Step | Class | Size | Use |
|---|---|---|---|
| Caption | `text-caption` | 10px | Badges, chips, dense meta text in tables |
| Small | `text-xs` | 12px | Labels, hints, secondary text, small buttons |
| Body | `text-sm` | 14px | Body text, form values, buttons |
| Large | `text-base`, `text-lg` | 16px, 18px | Emphasis, card titles |
| Heading | `text-xl`, `text-2xl` | 20px, 24px | Page headings, stat figures |

Do not use pixel sizes such as `text-[11px]`, including inside a `<style>` block. The serving stat figures (`lg:text-[32px]`) and the 16px iOS input are the only exceptions. Enhanced Readability raises `text-caption` and badge text to 12px. The off-ramp sizes still on the chat page (11px inline; 13px and 14.4px, written as `0.8125rem` and `0.9rem`) and the two display sizes (30 and 36px) are tracked in #4400.

## Buttons

- Text buttons are pills: `rounded-full`. Primary is `bg-neutral-900 text-white hover:bg-neutral-800`; secondary uses a border or `bg-neutral-100`.
- Use `hover:bg-neutral-800`, not `hover:bg-neutral-700`: only `bg-neutral-900` and `hover:bg-neutral-800` map to `--btn-primary*`, so 700 falls back to the palette and gives dark text on a dark hover.
- Icon-only buttons are `rounded-lg`.
- Small buttons that repeat in every row of a table or list are `rounded-lg`, so a row of actions reads as one unit.
- Focus rings are `focus:ring-2 focus:ring-offset-2` with `ring-neutral-900` or `ring-black`; those are the two ring utilities remapped to `--focus-ring-color` on the dashboard. The chat and login pages do not load that remap yet; the move is tracked in #4400.

Segmented items, menu rows, option cards and plain text links are not buttons for these rules.

## Choice controls

- Pick one of a few values: the `.segmented` control in `dashboard.css`. `ui.segmented` in `templates/components/ui.html` renders it for plain text items. The macro has no class parameter, so `segmented--sm` is added only in hand-written markup (`class="segmented segmented--sm"`). Items that need icons, `x-for` or extra attributes use the classes directly: `segmented__item` on each button and `segmented__item--active` on the selected one.
- On and off: the switch below. The track is 44x24, black when on and `bg-neutral-200` when off. Dark mode colors come from the `--switch-on` token. A switch that turns a dangerous capability on keeps `bg-red-600`, as in Settings → Auth & Info and Model settings → Trust remote code. The chat page's `.svg-allow-switch` is its own smaller control; moving it onto `--switch-on` is tracked in #4400.

```html
<button type="button" role="switch"
        :aria-checked="value ? 'true' : 'false'"
        :class="value ? 'bg-black' : 'bg-neutral-200'"
        class="relative w-11 h-6 flex-shrink-0 rounded-full transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-black">
    <span :class="value ? 'translate-x-5' : 'translate-x-0'"
          class="block w-5 h-5 bg-white rounded-full shadow-sm transform transition-transform duration-300 absolute top-0.5 left-0.5"></span>
</button>
```

## Badges

Status labels use the `.badge` class from `dashboard.css`: a 10px pill with fixed padding and a 1px border. Set the tone with color utilities, usually `bg-<tone>-50 text-<tone>-700 border-<tone>-200`. The class has zero specificity, so those utilities always win. Do not set padding, size or radius on a badge.

For green, amber, red and blue the tone utilities resolve through `status.css` to the role channels; use a step that has a channel there. Other hues use the palette directly.

## Corners

| Element | Class |
|---|---|
| Card (a top-level block), dialog | `rounded-2xl` |
| Panel or option card inside a card | `rounded-xl` |
| Input, select, textarea, dropdown menu, icon button, row button | `rounded-lg` |
| Text button, badge, switch | `rounded-full` |

Checkboxes, tooltips and inline code chips keep their small `rounded` or `rounded-md` corners. The benchmark tooltip is an 8px exception; the chat page keeps its own radii. The navbar's menus are `rounded-xl` and keep their own shape, like the rest of the navbar.
