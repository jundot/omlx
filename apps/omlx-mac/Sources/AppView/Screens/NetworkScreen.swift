// Phase 2 — Network.
//
// Process-wide outbound plumbing — applies to every HTTP call the server
// makes (HF, ModelScope, Sparkle, future). Two sections:
//   • Outbound proxies — `http_proxy` / `https_proxy` / `no_proxy`.
//   • TLS — `ca_bundle` (custom root CA path).
//
// Mirror endpoints (HF, MS) live on the Downloads tab instead — they're
// contextual to the source the user is downloading from, so the editor
// swaps with the active source. Network stays focused on settings that
// affect every outbound call regardless of source.
//
// All fields are free-text path/URL strings, so the screen uses the
// Storage / MCP pattern: edit freely, click Apply to commit, button stays
// disabled until the draft diverges from the last-loaded values.

import SwiftUI

struct NetworkScreen: View {
    @Environment(AppServices.self) private var services
    @State private var vm = NetworkScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ProxiesSection(vm: vm)
            TLSSection(vm: vm)

            HStack {
                Spacer()
                Button(String(localized: "network.button.apply",
                              defaultValue: "Apply",
                              comment: "Footer button on the Network screen that commits the edited proxy/TLS values to the server")) {
                    Task { await vm.save(client: services.client) }
                }
                .buttonStyle(.omlx(.primary))
                .disabled(!vm.hasPendingChanges || vm.isSaving)
            }
            .padding(.horizontal, 18)
            .padding(.top, 6)

            HintFooter(error: vm.lastError)
        }
        .task { await vm.load(client: services.client) }
    }
}

// MARK: - Proxies

private struct ProxiesSection: View {
    @Bindable var vm: NetworkScreenVM

    var body: some View {
        SectionHeader(
            String(localized: "network.section.proxies.title",
                   defaultValue: "Proxies",
                   comment: "Section header above the outbound proxy fields on the Network screen"),
            subtitle: String(localized: "network.section.proxies.subtitle",
                             defaultValue: "Outbound HTTP routing. Empty = no proxy. Applied via HTTP_PROXY / HTTPS_PROXY / NO_PROXY env vars.",
                             comment: "Subtitle under the Proxies section header explaining how the values are applied")
        )

        ListGroup {
            Row(label: String(localized: "network.row.http_proxy.label",
                              defaultValue: "HTTP proxy",
                              comment: "Row label for the HTTP_PROXY field on the Network screen")) {
                TextInput(
                    text: $vm.httpProxy,
                    placeholder: "http://proxy.local:8080",
                    mono: true,
                    width: 320
                )
            }
            Row(label: String(localized: "network.row.https_proxy.label",
                              defaultValue: "HTTPS proxy",
                              comment: "Row label for the HTTPS_PROXY field on the Network screen")) {
                TextInput(
                    text: $vm.httpsProxy,
                    placeholder: "http://proxy.local:8080",
                    mono: true,
                    width: 320
                )
            }
            Row(
                label: String(localized: "network.row.no_proxy.label",
                              defaultValue: "No proxy",
                              comment: "Row label for the NO_PROXY field on the Network screen"),
                sublabel: String(localized: "network.row.no_proxy.sub",
                                 defaultValue: "Comma-separated host/CIDR list to bypass the proxy.",
                                 comment: "Sublabel explaining the No proxy field format on the Network screen"),
                isLast: true
            ) {
                TextInput(
                    text: $vm.noProxy,
                    placeholder: "localhost,127.0.0.1,*.internal",
                    mono: true,
                    width: 320
                )
            }
        }
    }
}

// MARK: - TLS

private struct TLSSection: View {
    @Bindable var vm: NetworkScreenVM

    var body: some View {
        SectionHeader(
            String(localized: "network.section.tls.title",
                   defaultValue: "TLS",
                   comment: "Section header above the TLS / custom CA fields on the Network screen"),
            subtitle: String(localized: "network.section.tls.subtitle",
                             defaultValue: "Custom root CA for environments with TLS-inspecting proxies.",
                             comment: "Subtitle under the TLS section header on the Network screen")
        )

        ListGroup {
            Row(
                label: String(localized: "network.row.ca_bundle.label",
                              defaultValue: "CA bundle",
                              comment: "Row label for the custom CA bundle path field on the Network screen"),
                sublabel: String(localized: "network.row.ca_bundle.sub",
                                 defaultValue: "Absolute path to a PEM-encoded CA bundle. Empty = system trust store.",
                                 comment: "Sublabel explaining the CA bundle field on the Network screen"),
                isLast: true
            ) {
                TextInput(
                    text: $vm.caBundle,
                    placeholder: "/etc/ssl/certs/ca-bundle.pem",
                    mono: true,
                    width: 320
                )
            }
        }
    }
}

// MARK: - Hint footer

private struct HintFooter: View {
    let error: String?

    var body: some View {
        if let error {
            Text(error)
                .font(.omlxText(11))
                .foregroundStyle(.red)
                .padding(.horizontal, 18)
                .padding(.top, 8)
        }
    }
}
