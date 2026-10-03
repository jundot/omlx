/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/js/**/*.js",
  ],
  safelist: [
    "sm:grid-cols-2",  // dynamic :class in _modal_model_settings.html
    "bg-emerald-500", "text-white", "border-emerald-500",
    "bg-emerald-50", "text-emerald-700", "border-emerald-200", "hover:bg-emerald-100",
  ],
  theme: {
    extend: {
      fontFamily: {
        // System stack only, shared with the app (see tokens.json).
        sans: ['var(--font-sans)'],
      },
      colors: {
        surface: {
          DEFAULT: 'var(--bg-primary)',
          alt: 'var(--bg-secondary)',
          muted: 'var(--bg-tertiary)',
        },
        fg: {
          DEFAULT: 'var(--text-primary)',
          secondary: 'var(--text-secondary)',
          tertiary: 'var(--text-tertiary)',
        },
        line: {
          DEFAULT: 'var(--border-faint)',
          strong: 'var(--border-normal)',
        },
        // The console's tinted action: the same blue the buttons use, so
        // `bg-accent` in a page's markup and `.btn--primary` never disagree.
        accent: {
          DEFAULT: 'var(--accent-fill)',
          hover: 'var(--accent-fill-hover)',
          fg: 'var(--accent-fill-text)',
        },
        // `DEFAULT` is the destructive *text* colour; `fill` is the destructive
        // button's fill, the same token `.btn--destructive` paints with.
        danger: {
          DEFAULT: 'var(--text-danger)',
          bg: 'var(--bg-danger-hover)',
          fill: 'var(--btn-danger)',
          fg: 'var(--danger-fg)',
        },
        code: 'var(--code-bg)',
        // Apple system colours (tokens.json semantic.*), for status semantics.
        sys: {
          red: 'var(--sys-red)',
          orange: 'var(--sys-orange)',
          green: 'var(--sys-green)',
          blue: 'var(--sys-blue)',
        },
        // --- the palette the markup still paints with Tailwind's defaults ---
        // `white` and `black` deliberately keep Tailwind's literals: Tailwind
        // cannot attach an opacity modifier to a var() colour — it drops
        // `bg-white/80`, `bg-black/50`, `ring-black/10` … from the build
        // entirely — and those alpha surfaces are in the markup. The two
        // non-alpha cases are wired by companions instead: `.bg-white` and
        // `.bg-black` in components.css (the surfaces, which change with the
        // theme) and `textColor.white` below (the label on a fill), which is
        // how the one shared key serves both directions (tokens.json
        // `control.onFill`).
        // Background-oriented neutrals: what the *fill* of a surface converges
        // onto in the light appearance; the dark appearance already painted
        // these through dashboard.css's overrides, which now drop away where
        // this mapping declares the same value. Shades with no honest token
        // (300/500/700/950) stay on Tailwind's default — see the PR report.
        // 200 takes `surface.bgStrong`, not bgTertiary: in the light
        // appearance the two would collapse onto the same #ececee and the
        // hover step on a neutral-100 base (13 bench rows) would disappear;
        // bgStrong keeps today's #e5e5e5 there, and its dark value is
        // bgTertiary's, so the night pixels do not move either.
        neutral: {
          50: 'var(--bg-secondary)',
          100: 'var(--bg-tertiary)',
          200: 'var(--bg-strong)',
          800: 'var(--accent-fill-hover)',
          900: 'var(--btn-primary)',
        },
        // Status fills converge onto the console's palette: the label fills
        // are pixel-identical to the Tailwind -50 literals they replace, the
        // destructive fill is the console's own, and the dots/bars take the
        // system colours (tokens.json semantic.*: "status marks only").
        red: {
          50: 'var(--label-red)',
          500: 'var(--btn-danger)',
          200: 'color-mix(in srgb, var(--sys-red) 30%, transparent)',
        },
        amber: {
          50: 'var(--label-amber)',
          200: 'color-mix(in srgb, var(--sys-orange) 30%, transparent)',
          300: 'var(--sys-orange)',
          400: 'var(--sys-orange)',
        },
        green: {
          50: 'var(--label-green)',
          200: 'color-mix(in srgb, var(--sys-green) 30%, transparent)',
          400: 'var(--sys-green)',
          500: 'var(--sys-green)',
        },
        emerald: {
          50: 'var(--label-emerald)',
          200: 'color-mix(in srgb, var(--sys-green) 30%, transparent)',
          500: 'var(--sys-green)',
        },
      },
      // The one small companion to the shared `colors` namespace: `text-white`
      // is not the surface — it is the label that reads on a fill in both
      // appearances (70 uses on badges, tooltips and segments), while
      // `bg-white` (152 uses) is the surface and resolves through the
      // `.bg-white` rule in components.css. A single mapping cannot serve
      // both, so text resolves here and surfaces there; the contextual flip
      // inside an inverted dark button stays a rule in dashboard.css, because
      // no key can say "only inside button.bg-neutral-900".
      textColor: {
        white: 'var(--on-fill)',
        // Text greys converge onto the three text tokens: the same mapping
        // dashboard.css's dark overrides already declared per utility, now
        // said once at the resolver. 300 stays off the list on purpose — its
        // light value is the faint decorative grey (separators, disabled
        // steps) and no token has that pair; its dark override is kept.
        neutral: {
          // 100/200 are compiled only because base.html's enhanced-readability
          // selector block names them; no element carries them, and they keep
          // Tailwind's literals so the pair cannot inherit a *fill* token.
          100: '#f5f5f5',
          200: '#e5e5e5',
          400: 'var(--text-tertiary)',
          500: 'var(--text-tertiary)',
          600: 'var(--text-secondary)',
          700: 'var(--text-secondary)',
          800: 'var(--text-primary)',
          900: 'var(--text-primary)',
        },
        // Status text: the tinted families land on the step that reads on
        // their own tint (badge.*), the bright marks on the system colours.
        red: {
          400: 'var(--badge-red-fg)',
          500: 'var(--badge-red-fg)',
          600: 'var(--badge-red-fg)',
          700: 'var(--badge-red-fg)',
          800: 'var(--badge-red-fg)',
        },
        amber: {
          400: 'var(--sys-orange)',
          500: 'var(--sys-orange)',
          600: 'var(--badge-orange-fg)',
          700: 'var(--badge-orange-fg)',
          800: 'var(--badge-orange-fg)',
          900: 'var(--badge-orange-fg)',
        },
        green: {
          300: 'var(--sys-green)',
          400: 'var(--sys-green)',
          500: 'var(--sys-green)',
          600: 'var(--badge-green-fg)',
          700: 'var(--badge-green-fg)',
          800: 'var(--badge-green-fg)',
          900: 'var(--badge-green-fg)',
        },
        emerald: {
          600: 'var(--label-emerald-fg)',
          700: 'var(--label-emerald-fg)',
        },
        blue: {
          500: 'var(--badge-blue-fg)',
          700: 'var(--badge-blue-fg)',
          900: 'var(--badge-blue-fg)',
        },
      },
      borderColor: {
        // Hairlines: the neutral borders converge onto the border pair (the
        // dark values are exactly what dashboard.css declared). 50 keeps the
        // fill it inherits from `colors`. 800/900 stay literal because no
        // token has their same-value pair in both appearances and the fabric
        // and the number input would move in the dark appearance otherwise.
        // 300/400 take the border pair / tertiary step the overrides named.
        neutral: {
          100: 'var(--border-faint)',
          200: 'var(--border-faint)',
          300: 'var(--border-normal)',
          400: 'var(--text-tertiary)',
          800: '#262626',
          900: '#171717',
        },
      },
      divideColor: {
        neutral: {
          100: 'var(--border-faint)',
          200: 'var(--border-faint)',
        },
      },
      ringColor: {
        // Rings keep today's literals, deliberately: `ring-neutral-900` is two
        // things at once — the focus ring (whose dark step is
        // `--border-normal`, a dashboard.css override that stays) and a
        // decorative ring on the cluster cards (which has no override and
        // must stay #171717 at night) — and one key cannot be both; mapping
        // it would light the cluster cards with a white ring in the dark
        // appearance. Same story for `ring-red-500` and `ring-black` (whose
        // /10 alpha variant Tailwind would drop if the key became a var()).
        // The literals are the pixels that render today.
        neutral: {
          900: '#171717',
        },
        red: {
          500: '#ef4444',
        },
      },
      placeholderColor: {
        neutral: {
          // 100/200 are compiled only because base.html's enhanced-readability
          // selector block names them; no element carries them, and they keep
          // Tailwind's literals so the pair cannot inherit a *fill* token.
          100: '#f5f5f5',
          200: '#e5e5e5',
          400: 'var(--text-tertiary)',
        },
      },
      borderRadius: {
        // Mapped by pixel equality, never by name: rounded-md (6px) wears the
        // token called sm, rounded-lg (8px) the token called md, rounded-xl
        // (12px) lg, and rounded-full (9999px) pill (999px) — the same rendered
        // shape at every size the console draws. Recorded in tokens.json.
        DEFAULT: 'var(--radius-4)',
        sm: 'var(--radius-2)',
        md: 'var(--radius-sm)',
        lg: 'var(--radius-md)',
        xl: 'var(--radius-lg)',
        '2xl': 'var(--radius-16)',
        '3xl': 'var(--radius-24)',
        full: 'var(--radius-pill)',
      },
      spacing: {
        // Every step the markup uses, pointing at the token with equal
        // pixels: Tailwind's number is the index × 4px, the token's name is
        // its own value, so `p-6` (24px) wears `--space-5` and `w-64` (256px)
        // wears `--space-256`. The grid keys never move (tokens.json).
        0: 'var(--space-0)',
        '0.5': 'var(--space-2px)',
        1: 'var(--space-1)',
        '1.5': 'var(--space-6px)',
        2: 'var(--space-2)',
        '2.5': 'var(--space-10px)',
        3: 'var(--space-3)',
        '3.5': 'var(--space-14)',
        4: 'var(--space-4)',
        5: 'var(--space-20)',
        6: 'var(--space-5)',
        7: 'var(--space-28)',
        8: 'var(--space-6)',
        9: 'var(--space-36)',
        10: 'var(--space-40)',
        11: 'var(--space-44)',
        12: 'var(--space-8)',
        14: 'var(--space-56)',
        16: 'var(--space-10)',
        20: 'var(--space-80)',
        24: 'var(--space-96)',
        28: 'var(--space-112)',
        32: 'var(--space-128)',
        36: 'var(--space-144)',
        40: 'var(--space-160)',
        44: 'var(--space-176)',
        48: 'var(--space-192)',
        52: 'var(--space-208)',
        56: 'var(--space-224)',
        60: 'var(--space-240)',
        64: 'var(--space-256)',
        80: 'var(--space-320)',
        96: 'var(--space-384)',
      },
      boxShadow: {
        // The elevation tokens, so a panel in a template names its depth
        // rather than inventing a Tailwind shadow. The six below are
        // Tailwind's own values copied verbatim — pixels may not move — and
        // the dark overrides that give a few of them a heavier night weight
        // stay in dashboard.css, because one value per key cannot be two.
        card: 'var(--shadow-card)',
        popover: 'var(--shadow-popover)',
        DEFAULT: 'var(--shadow-default)',
        sm: 'var(--shadow-sm)',
        md: 'var(--shadow-md)',
        lg: 'var(--shadow-lg)',
        xl: 'var(--shadow-xl)',
        '2xl': 'var(--shadow-2xl)',
      },
      opacity: {
        0: 'var(--opacity-0)',
        25: 'var(--opacity-25)',
        40: 'var(--opacity-40)',
        50: 'var(--opacity-50)',
        60: 'var(--opacity-60)',
        70: 'var(--opacity-70)',
        75: 'var(--opacity-75)',
        80: 'var(--opacity-80)',
        100: 'var(--opacity-100)',
      },
      letterSpacing: {
        tight: 'var(--tracking-tight)',
        wide: 'var(--tracking-wide)',
        wider: 'var(--tracking-wider)',
        widest: 'var(--tracking-widest)',
      },
      lineHeight: {
        // The four ratios the markup uses, at their exact current values — a
        // line-height is a typography value and may not move anywhere.
        none: 'var(--lh-none)',
        tight: 'var(--lh-tight)',
        snug: 'var(--lh-snug)',
        relaxed: 'var(--lh-relaxed)',
      },
      fontWeight: {
        thin: 'var(--fw-100)',
        extralight: 'var(--fw-200)',
        light: 'var(--fw-300)',
        normal: 'var(--fw-400)',
        medium: 'var(--fw-500)',
        semibold: 'var(--fw-600)',
        bold: 'var(--fw-700)',
        extrabold: 'var(--fw-800)',
        // `font-black` (900) stays Tailwind's default: the group tops out at
        // 800 and nothing in the markup uses black.
      },
      transitionDuration: {
        // 200ms is `normal`, the group's own key; 300 gets its own key so
        // `slow` stays paper-only (tokens.json console.duration).
        100: 'var(--duration-100)',
        150: 'var(--duration-150)',
        200: 'var(--duration-normal)',
        300: 'var(--duration-300)',
        500: 'var(--duration-500)',
        700: 'var(--duration-700)',
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.5s ease-out forwards',
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'none' },
        }
      }
    },
    // Exactly six type levels (tokens.json typography.scale). Replacing rather
    // than extending keeps off-scale sizes out of the utility set entirely;
    // the legacy Tailwind names stay as aliases so existing markup keeps
    // working while snapping onto the scale.
    fontSize: {
      xs: ['var(--fs-aux)', { lineHeight: 'var(--lh-aux)' }],
      sm: ['var(--fs-body)', { lineHeight: 'var(--lh-body)' }],
      base: ['var(--fs-emphasis)', { lineHeight: 'var(--lh-emphasis)' }],
      lg: ['var(--fs-emphasis)', { lineHeight: 'var(--lh-emphasis)' }],
      xl: ['var(--fs-section)', { lineHeight: 'var(--lh-section)' }],
      '2xl': ['var(--fs-page)', { lineHeight: 'var(--lh-page)' }],
      '3xl': ['var(--fs-kpi)', { lineHeight: 'var(--lh-kpi)' }],
      '4xl': ['var(--fs-kpi)', { lineHeight: 'var(--lh-kpi)' }],
      aux: ['var(--fs-aux)', { lineHeight: 'var(--lh-aux)' }],
      body: ['var(--fs-body)', { lineHeight: 'var(--lh-body)' }],
      emphasis: ['var(--fs-emphasis)', { lineHeight: 'var(--lh-emphasis)' }],
      section: ['var(--fs-section)', { lineHeight: 'var(--lh-section)' }],
      page: ['var(--fs-page)', { lineHeight: 'var(--lh-page)' }],
      kpi: ['var(--fs-kpi)', { lineHeight: 'var(--lh-kpi)' }],
    },
  },
  plugins: [],
}
