# Web UI design rules

These rules cover the web UI in `apps/omlx-web/omlx_web`. Colors and type steps apply to every page, including the chat page and the top navbar. The button, choice control, badge and corner rules cover login and the dashboard; the chat page and the navbar keep their own shapes. Use them when you add or change a template, and rebuild the CSS afterwards with `cd apps/omlx-web && python build_css.py`.

## Colors

Every color is defined in `apps/omlx-web/omlx_web/static/css/theme.css`, which `base.html` loads before `tailwind.css` and the page stylesheets.

- Palette entries are named `--palette-<family>-<step>` and hold `R G B` channels, so both `rgb(var(--palette-blue-500))` and `rgb(var(--palette-blue-500) / 0.2)` work.
- Semantic tokens name a role and point at the palette: `--bg-primary`, `--bg-secondary`, `--bg-tertiary`, `--text-primary` ... `--text-muted`, `--border-faint`, `--border-normal`, `--code-bg`, `--link-color`, `--btn-primary*`, `--text-danger`, `--bg-danger-hover`, `--timeline-accent`, `--focus-ring-color`.
- `[data-theme="dark"]` in the same file gives the dark values of the semantic tokens. `dashboard.css` maps light utilities such as `.bg-white` or `.text-neutral-900` to dark values under the same selector.
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
- `fabric-ink-*`, `fabric-lime-*`: the cluster fabric diagram only.

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
- On and off: the switch below. The track is 44x24, black when on and `bg-neutral-200` when off. Dark mode colors come from `dashboard.css`.

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
