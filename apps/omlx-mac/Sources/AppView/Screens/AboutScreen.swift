// PR 9 — About.
//
// Build info + license + credits + project links. The Updates section does
// NOT live here — design v2 moved it onto Status (PR 7). This screen is
// intentionally static; opening links bounces to the default browser.

import SwiftUI
import AppKit

struct AboutScreen: View {
    @Environment(\.omlxTheme) private var theme

    private var version: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0.0"
    }
    private var build: String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "—"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HeroCard(version: version, build: build)
            ProjectSection()
            LicenseSection()
            CreditsSection()
        }
    }
}

// MARK: - Hero

private struct HeroCard: View {
    let version: String
    let build: String

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 16) {
            // Match the ServerHeroCard / StatusScreen hero — same rounded
            // omlx mark as the Dock icon and README hero, light/dark
            // variants ship with the AppLogo imageset.
            //
            // The AppLogo SVG embeds a 10pt transparent margin inside its
            // 160pt viewBox (content occupies 140pt). Frame is scaled by
            // 160/140 so the visible rounded-square reads at the same ~60pt
            // size as the previous Squircle placeholder.
            Image("AppLogo")
                .resizable()
                .interpolation(.high)
                .frame(width: 69, height: 69)
            VStack(alignment: .leading, spacing: 4) {
                Text("oMLX")
                    .font(.omlxText(22, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text(String(localized: "about.hero.tagline",
                            defaultValue: "Local AI, no more waiting on your Mac.",
                            comment: "Tagline shown under the oMLX product name on the About screen hero card"))
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textSecondary)
                Text(String(localized: "about.hero.version",
                            defaultValue: "Version \(version) · build \(build)",
                            comment: "Version + build line on the About screen hero card; placeholders are the bundle short version string and bundle version"))
                    .font(.omlxMono(11))
                    .foregroundStyle(theme.textTertiary)
            }
            Spacer(minLength: 8)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(theme.groupBg)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
        .padding(.horizontal, 14)
        .padding(.bottom, 14)
    }
}

// MARK: - Project section

private struct ProjectSection: View {
    var body: some View {
        SectionHeader(String(localized: "about.section.project",
                             defaultValue: "Project",
                             comment: "Section header above the project links list on the About screen"))

        ListGroup {
            LinkRow(
                label: String(localized: "about.project.github.label",
                              defaultValue: "GitHub Repository",
                              comment: "About screen link row label pointing to the oMLX GitHub repo"),
                sublabel: String(localized: "about.project.github.sub",
                                 defaultValue: "Source, issues, and roadmap",
                                 comment: "Sublabel under the GitHub Repository link on the About screen"),
                icon: "chevron.left.forwardslash.chevron.right",
                url: URL(string: "https://github.com/jundot/omlx")!
            )
            LinkRow(
                label: String(localized: "about.project.releases.label",
                              defaultValue: "Releases",
                              comment: "About screen link row label pointing to GitHub releases"),
                sublabel: String(localized: "about.project.releases.sub",
                                 defaultValue: "Download the latest CLI and macOS app",
                                 comment: "Sublabel under the Releases link on the About screen"),
                icon: "shippingbox",
                url: URL(string: "https://github.com/jundot/omlx/releases")!
            )
            LinkRow(
                label: String(localized: "about.project.docs.label",
                              defaultValue: "Documentation",
                              comment: "About screen link row label pointing to product documentation"),
                sublabel: String(localized: "about.project.docs.sub",
                                 defaultValue: "Setup, model management, integrations",
                                 comment: "Sublabel under the Documentation link on the About screen"),
                icon: "book.closed",
                url: URL(string: "https://omlx.app/docs")!
            )
            LinkRow(
                label: String(localized: "about.project.issue.label",
                              defaultValue: "Report an Issue",
                              comment: "About screen link row label pointing to the GitHub new-issue form"),
                sublabel: String(localized: "about.project.issue.sub",
                                 defaultValue: "Bugs and feature requests on GitHub",
                                 comment: "Sublabel under the Report an Issue link on the About screen"),
                icon: "exclamationmark.bubble",
                url: URL(string: "https://github.com/jundot/omlx/issues/new")!,
                isLast: true
            )
        }
    }
}

private struct LinkRow: View {
    let label: String
    let sublabel: String
    let icon: String
    let url: URL
    var isLast: Bool = false

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        FreeRow(isLast: isLast) {
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.system(size: 14))
                    .foregroundStyle(theme.textSecondary)
                    .frame(width: 18, alignment: .center)
                VStack(alignment: .leading, spacing: 2) {
                    Text(label)
                        .font(.omlxText(13, weight: .medium))
                        .foregroundStyle(theme.text)
                    Text(sublabel)
                        .font(.omlxText(11))
                        .foregroundStyle(theme.textSecondary)
                }
                Spacer(minLength: 8)
                Button {
                    NSWorkspace.shared.open(url)
                } label: {
                    Label(String(localized: "common.open",
                                 defaultValue: "Open",
                                 comment: "Button label that opens the linked URL in the default browser"),
                          systemImage: "arrow.up.right.square")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.omlx(.normal, size: .small))
            }
        }
    }
}

// MARK: - License

private struct LicenseSection: View {
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(String(localized: "about.section.license",
                             defaultValue: "License",
                             comment: "Section header above the license block on the About screen"))

        ListGroup {
            FreeRow(isLast: true) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 6) {
                        Image(systemName: "scale.3d")
                            .font(.system(size: 13))
                            .foregroundStyle(theme.textSecondary)
                        Text(String(localized: "about.license.name",
                                    defaultValue: "Apache License 2.0",
                                    comment: "Name of the open-source license shown on the About screen"))
                            .font(.omlxText(13, weight: .medium))
                            .foregroundStyle(theme.text)
                    }
                    Text(String(localized: "about.license.notice",
                                defaultValue: "Copyright © oMLX contributors. Licensed under the Apache License, Version 2.0. See the LICENSE file in the repository for the full text.",
                                comment: "Copyright + license notice paragraph on the About screen"))
                        .font(.omlxText(11.5))
                        .foregroundStyle(theme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

// MARK: - Credits

private struct CreditsSection: View {
    @Environment(\.omlxTheme) private var theme

    private struct Credit: Identifiable {
        let id = UUID()
        let name: String
        let role: String
        let url: URL
    }

    private let credits: [Credit] = [
        Credit(
            name: "MLX",
            role: String(localized: "about.credits.mlx.role",
                         defaultValue: "Apple's array framework — the engine behind every model",
                         comment: "About screen Credits row: role/description for the MLX project"),
            url: URL(string: "https://github.com/ml-explore/mlx")!
        ),
        Credit(
            name: "mlx-lm",
            role: String(localized: "about.credits.mlx_lm.role",
                         defaultValue: "Language-model execution + fine-tuning on MLX",
                         comment: "About screen Credits row: role/description for the mlx-lm project"),
            url: URL(string: "https://github.com/ml-explore/mlx-lm")!
        ),
        Credit(
            name: "mlx-vlm",
            role: String(localized: "about.credits.mlx_vlm.role",
                         defaultValue: "Vision-language models on MLX",
                         comment: "About screen Credits row: role/description for the mlx-vlm project"),
            url: URL(string: "https://github.com/Blaizzy/mlx-vlm")!
        ),
        Credit(
            name: "mlx-embeddings",
            role: String(localized: "about.credits.mlx_embeddings.role",
                         defaultValue: "Embedding + reranker models on MLX",
                         comment: "About screen Credits row: role/description for the mlx-embeddings project"),
            url: URL(string: "https://github.com/Blaizzy/mlx-embeddings")!
        ),
        Credit(
            name: "mlx-audio",
            role: String(localized: "about.credits.mlx_audio.role",
                         defaultValue: "Audio (STT / TTS / STS) models on MLX",
                         comment: "About screen Credits row: role/description for the mlx-audio project"),
            url: URL(string: "https://github.com/Blaizzy/mlx-audio")!
        ),
    ]

    var body: some View {
        SectionHeader(String(localized: "about.section.built_on",
                             defaultValue: "Built On",
                             comment: "Section header above the credits list on the About screen"))

        ListGroup {
            ForEach(Array(credits.enumerated()), id: \.element.id) { idx, credit in
                FreeRow(isLast: idx == credits.count - 1) {
                    HStack(spacing: 10) {
                        Squircle(systemSymbol: "cpu",
                                 size: 26,
                                 gradient: SquircleGradient.models)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(credit.name)
                                .font(.omlxText(13, weight: .medium))
                                .foregroundStyle(theme.text)
                            Text(credit.role)
                                .font(.omlxText(11))
                                .foregroundStyle(theme.textSecondary)
                                .lineLimit(2)
                        }
                        Spacer(minLength: 8)
                        Button {
                            NSWorkspace.shared.open(credit.url)
                        } label: {
                            Image(systemName: "arrow.up.right.square")
                                .font(.system(size: 12))
                        }
                        .buttonStyle(.omlx(.plain, size: .small))
                    }
                }
            }
        }
    }
}
