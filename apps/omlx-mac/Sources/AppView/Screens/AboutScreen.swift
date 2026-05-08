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
            Squircle(gradient: SquircleGradient.server, size: 60) {
                Text("oM")
                    .font(.omlxText(26, weight: .heavy))
                    .kerning(-0.6)
                    .foregroundStyle(.white)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("oMLX")
                    .font(.omlxText(22, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text("Native macOS app for the oMLX server")
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textSecondary)
                Text("Version \(version) · build \(build)")
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
        SectionHeader("Project")

        ListGroup {
            LinkRow(
                label: "GitHub Repository",
                sublabel: "Source, issues, and roadmap",
                icon: "chevron.left.forwardslash.chevron.right",
                url: URL(string: "https://github.com/jundot/omlx")!
            )
            LinkRow(
                label: "Releases",
                sublabel: "Download the latest CLI and macOS app",
                icon: "shippingbox",
                url: URL(string: "https://github.com/jundot/omlx/releases")!
            )
            LinkRow(
                label: "Documentation",
                sublabel: "Setup, model management, integrations",
                icon: "book.closed",
                url: URL(string: "https://omlx.app/docs")!
            )
            LinkRow(
                label: "Report an Issue",
                sublabel: "Bugs and feature requests on GitHub",
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
                    Label("Open", systemImage: "arrow.up.right.square")
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
        SectionHeader("License")

        ListGroup {
            FreeRow(isLast: true) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 6) {
                        Image(systemName: "scale.3d")
                            .font(.system(size: 13))
                            .foregroundStyle(theme.textSecondary)
                        Text("Apache License 2.0")
                            .font(.omlxText(13, weight: .medium))
                            .foregroundStyle(theme.text)
                    }
                    Text("Copyright © oMLX contributors. Licensed under the Apache License, Version 2.0. See the LICENSE file in the repository for the full text.")
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
            role: "Apple's array framework — the engine behind every model",
            url: URL(string: "https://github.com/ml-explore/mlx")!
        ),
        Credit(
            name: "mlx-lm",
            role: "Language-model execution + fine-tuning on MLX",
            url: URL(string: "https://github.com/ml-explore/mlx-lm")!
        ),
        Credit(
            name: "mlx-vlm",
            role: "Vision-language models on MLX",
            url: URL(string: "https://github.com/Blaizzy/mlx-vlm")!
        ),
        Credit(
            name: "mlx-embeddings",
            role: "Embedding + reranker models on MLX",
            url: URL(string: "https://github.com/Blaizzy/mlx-embeddings")!
        ),
        Credit(
            name: "mlx-audio",
            role: "Audio (STT / TTS / STS) models on MLX",
            url: URL(string: "https://github.com/Blaizzy/mlx-audio")!
        ),
    ]

    var body: some View {
        SectionHeader("Built On")

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
