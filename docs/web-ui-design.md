# Web UI design rules

These rules cover the web UI in `apps/omlx-web/omlx_web`. Colors and type steps apply to every page, including the chat page and the top navbar. The button, choice control, badge and corner rules cover login and the dashboard; the chat page and the navbar keep their own shapes. Use them when you add or change a template, and rebuild the CSS afterwards with `cd apps/omlx-web && python build_css.py`.

## Colors

The palette and the light semantic colors are defined in `apps/omlx-web/omlx_web/static/css/theme.css`, which `base.html` loads before `tailwind.css`; `dashboard.css` supplies the dark values of the neutral utility channels.

- Palette entries are named `--palette-<family>-<step>` and hold `R G B` channels, so both `rgb(var(--palette-blue-500))` and `rgb(var(--palette-blue-500) / 0.2)` work.
- Semantic tokens name a role and point at the palette: `--bg-primary`, `--bg-secondary`, `--bg-tertiary`, `--text-primary` ... `--text-muted`, `--border-faint`, `--border-normal`, `--code-bg`, `--link-color`, `--btn-primary*`, `--switch-on`, `--text-danger`, `--bg-danger-hover`, `--timeline-accent`, `--focus-ring-color`. The status roles are `--success`, `--warning`, `--danger` and `--info`, each with `-soft`, `-line`, `-text` shades; `status.css` maps the badge/notice/dot utilities to them on every page.
- `[data-theme="dark"]` in the same file gives the dark values of the semantic tokens. The neutral utilities (`bg-white`, `text-neutral-*`, `border-neutral-*`) resolve through the `*-scale-*` channels: light values in `theme.css`, dark values in `dashboard.css`.
- `tailwind.config.js` resolves Tailwind's color utilities through the palette, so `bg-neutral-100` and `bg-neutral-100/50` read `theme.css` too.

Where to use what:

| Place | Use |
|---|---|
| Templates | Tailwind color utilities (`text-neutral-500`, `bg-amber-50`) |
| Stylesheets | Semantic tokens, or palette entries for one-off marks |
| Styles built in JS | `rgb(var(--palette-x) / alpha)` |

Do not write hex or `rgb()` literals outside `theme.css`. The one exception is the Enhanced Readability block in `base.html`: its `[style*="color: #..."]` selectors match inline style text, so they keep the literal hex.

Families outside Tailwind's palette:

- `night-*`: dark mode surfaces and text.
- `signal-red-*`: the readability red.
- `mid-gray`: empty cells in the usage strip.

## Custom theme

`ui.custom_theme` stores per-theme token overrides (`{"light": {"bg-primary": "#..."}, "dark": {...}}`), and `static/js/custom_theme.js` paints them before first paint. That file owns the token list and the contrast rules: the Theme Settings editor disables Apply, names the failing pairs and outlines the fields to change when primary/secondary text, links, button text (also on hover) or status badge text fall below 4.5:1 (WCAG 1.4.3 AA), or tertiary text, danger text, the focus ring, the switch or the accent fall below 3:1 (WCAG 1.4.11). Muted text and borders have no floor. The server only checks that each value is `#rrggbb`.

### How the dark draft is generated

**Generate dark from light** fills the dark column from the light one in OKLCH (`darkFromLight` in `custom_theme.js`); it only edits the draft, so Apply still runs the contrast check. The constants are calibrated so the stock light palette lands within about 0.03 OKLab ΔE of the stock dark one (text-muted is dimmer, about 0.06).

1. Neutrals (surfaces, text, borders, controls, code background) invert their lightness, `1 - L`, through a piecewise-linear curve (0→0.23, 0.13→0.44, 0.3→0.62, 0.55→0.72, 0.8→0.95, 1→1), so the light order is kept and white lands near the stock dark surface.
2. Neutrals keep their hue, with chroma capped at 0.012 so tints stay subtle; the switch ON track stops at L 0.72 because its knob stays white.
3. Status fills keep their color unless darker than L 0.62; links, the timeline accent and danger text are lifted to at least L 0.71, like stock blue-600 → blue-400.
4. A color outside sRGB keeps its lightness and hue and loses chroma until it fits.
5. A foreground below its contrast floor on any of its backgrounds moves its lightness away from them in 0.01 steps until it passes.

The Preview header's Save as image button writes both modes, every token's English name and exact hex, and the same palette as JSON in an `omlx-theme` `tEXt` chunk. The image is English-only so it can be handed to an agent or shared outside the dashboard. `theme_preview.js` draws the preview card for both: the Preview canvases in the UI language, the image with its fixed English labels, so the two never drift apart.

## Type

| Step | Class | Size | Use |
|---|---|---|---|
| Caption | `text-caption` | 10px | Badges, chips, dense meta text in tables |
| Small | `text-xs` | 12px | Labels, hints, secondary text, small buttons |
| Body | `text-sm` | 14px | Body text, form values, buttons |
| Large | `text-base`, `text-lg` | 16px, 18px | Emphasis, card titles |
| Heading | `text-xl`, `text-2xl` | 20px, 24px | Page headings, stat figures |

Do not use pixel sizes such as `text-[11px]`. The serving stat figures (`lg:text-[32px]`) are the only exception. Enhanced Readability raises `text-caption` and badge text to 12px.

## Buttons

- Text buttons are pills: `rounded-full`. Primary is `bg-neutral-900 text-white hover:bg-neutral-800`; secondary uses a border or `bg-neutral-100`.
- Icon-only buttons are `rounded-lg`.
- Small buttons that repeat in every row of a table or list are `rounded-lg`, so a row of actions reads as one unit.

Segmented items, menu rows, option cards and plain text links are not buttons for these rules.

## Choice controls

- Pick one of a few values: the `.segmented` control in `dashboard.css`. `ui.segmented` in `templates/components/ui.html` renders it for plain text items. Inside a card or a table row, add `segmented--sm`. Items that need icons, `x-for` or extra attributes use the classes directly: `segmented__item` on each button and `segmented__item--active` on the selected one.
- On and off: the switch below. The track is 44x24, black when on and `bg-neutral-200` when off. Dark mode colors come from the `--switch-on` token.

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

## Corners

| Element | Class |
|---|---|
| Card (a top-level block), dialog | `rounded-2xl` |
| Panel or option card inside a card | `rounded-xl` |
| Input, select, textarea, dropdown menu, icon button, row button | `rounded-lg` |
| Text button, badge, switch | `rounded-full` |

Checkboxes, tooltips and inline code chips keep their small `rounded` or `rounded-md` corners.
