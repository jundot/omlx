import SwiftUI

/// On-screen-only historical polling, independent of live serving stats.
struct UsageHistoryView: View {
    @Environment(AppServices.self) private var services
    @State private var period = "today"
    @State private var model = ""
    @State private var models: [String] = []
    @State private var data: UsageHistoryDTO?
    @State private var error: String?

    @State private var peak: Double = 1

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader(String(localized: "status.usage.heading",
                                 defaultValue: "Usage History",
                                 comment: "Section header for the local usage history panel"),
                          subtitle: String(localized: "status.usage.subtitle",
                                           defaultValue: "Local aggregates · server local time · 400-day retention",
                                           comment: "Subtitle under the usage history header describing data scope"))
            HStack {
                Picker(String(localized: "status.usage.range",
                              defaultValue: "Time range",
                              comment: "Picker label for the usage history time range"),
                       selection: $period) {
                    Text(String(localized: "status.usage.range.today",
                                defaultValue: "Today",
                                comment: "Usage history range option")).tag("today")
                    Text(String(localized: "status.usage.range.yesterday",
                                defaultValue: "Yesterday",
                                comment: "Usage history range option")).tag("yesterday")
                    Text(String(localized: "status.usage.range.7d",
                                defaultValue: "7 Days",
                                comment: "Usage history range option")).tag("7d")
                    Text(String(localized: "status.usage.range.30d",
                                defaultValue: "30 Days",
                                comment: "Usage history range option")).tag("30d")
                    Text(String(localized: "status.usage.range.90d",
                                defaultValue: "90 Days",
                                comment: "Usage history range option")).tag("90d")
                    Text(String(localized: "status.usage.range.month",
                                defaultValue: "This Month",
                                comment: "Usage history range option for the current calendar month")).tag("month")
                }
                Picker(String(localized: "status.usage.model",
                              defaultValue: "Model",
                              comment: "Picker label for the usage history model filter"),
                       selection: $model) {
                    Text(String(localized: "status.usage.all_models",
                                defaultValue: "All Models",
                                comment: "Usage history model filter option that includes every model")).tag("")
                    ForEach(models, id: \.self) { Text($0).tag($0) }
                }
            }
            .padding(.horizontal, 18)
            if let error {
                Text(error).font(.omlxText(12)).foregroundStyle(.secondary)
                    .padding(.horizontal, 18)
            }
            if let data {
                totals(data.totals)
                if data.totals.requests == 0 {
                    Text(String(localized: "status.usage.empty",
                                defaultValue: "No recorded usage in this range. History begins after upgrading.",
                                comment: "Shown when the selected usage history range has no recorded requests"))
                        .font(.omlxText(12)).foregroundStyle(.secondary)
                        .padding(.horizontal, 18)
                }
                ListGroup {
                    ForEach(data.models, id: \.modelId) { row in
                        Row(label: row.modelId ?? "", sublabel: detail(row)) {
                            VStack(alignment: .trailing, spacing: 3) {
                                Text(String(localized: "status.usage.row.tokens",
                                            defaultValue: "\(compact(row.totalTokens)) tokens",
                                            comment: "Per-model total token count; placeholder is a compact number"))
                                Text(String(localized: "status.usage.row.requests_speed",
                                            defaultValue: "\(row.requests) requests · \(speed(row.generationTps)) tok/s",
                                            comment: "Per-model request count and output speed; placeholders are a count and a formatted tokens-per-second value"))
                                    .foregroundStyle(.secondary)
                            }.font(.omlxMono(11))
                        }
                    }
                }
                heatmap(data.heatmap)
            } else if error == nil {
                ProgressView().padding(.horizontal, 18)
            }
        }
        .onChange(of: period) { _, _ in model = "" }
        .task(id: "\(period):\(model)") {
            data = nil
            while !Task.isCancelled {
                do {
                    let result = try await services.client.getUsage(range: period, model: model)
                    try Task.checkCancellation()
                    data = result
                    peak = Double(max(1, result.heatmap.flatMap(\.tokens).max() ?? 1))
                    if model.isEmpty { models = result.models.compactMap(\.modelId) }
                    error = result.available && result.droppedRequests == 0 ? nil :
                        String(localized: "status.usage.delayed",
                               defaultValue: "History may be incomplete: storage is delayed or the pending buffer overflowed.",
                               comment: "Warning shown when usage history storage is degraded but data is still displayed")
                } catch {
                    if Task.isCancelled { return }
                    data = nil
                    self.error = String(localized: "status.usage.unavailable",
                                        defaultValue: "Usage history is unavailable. Inference continues normally.",
                                        comment: "Error shown when the usage history endpoint cannot be reached")
                }
                do { try await Task.sleep(for: .seconds(15)) } catch { return }
            }
        }
    }

    private func totals(_ value: UsageHistoryDTO.UsageTotalsDTO) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 130))], spacing: 10) {
            tile(String(localized: "status.usage.tile.requests",
                        defaultValue: "Requests",
                        comment: "Usage history tile label for request count"),
                 value.requests)
            tile(String(localized: "status.usage.tile.total",
                        defaultValue: "Total Tokens",
                        comment: "Usage history tile label for prompt plus output tokens"),
                 value.totalTokens)
            tile(String(localized: "status.usage.tile.prompt",
                        defaultValue: "Prompt Tokens",
                        comment: "Usage history tile label for prompt tokens"),
                 value.promptTokens)
            tile(String(localized: "status.usage.tile.output",
                        defaultValue: "Output Tokens",
                        comment: "Usage history tile label for generated tokens"),
                 value.completionTokens)
            tile(String(localized: "status.usage.tile.cached",
                        defaultValue: "Cached Tokens",
                        comment: "Usage history tile label for prompt tokens served from cache"),
                 value.cachedTokens)
        }.padding(.horizontal, 18)
    }

    private func tile(_ label: String, _ value: Int) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label).font(.omlxText(11)).foregroundStyle(.secondary)
            Text(compact(value)).font(.omlxMono(20))
        }
        .frame(maxWidth: .infinity, alignment: .leading).padding(12)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
        .help(value.formatted())
    }

    private func heatmap(_ days: [UsageHistoryDTO.UsageDayDTO]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(String(localized: "status.usage.heatmap.title",
                        defaultValue: "Token Usage by Day and Hour",
                        comment: "Heading above the usage history day/hour heatmap"))
                .font(.omlxText(13))
            Text(String(localized: "status.usage.heatmap.note",
                        defaultValue: "00–23 hours · darker green means more tokens · repeated DST hours combine",
                        comment: "Legend text under the usage heatmap heading"))
                .font(.omlxText(11)).foregroundStyle(.secondary)
            HStack(spacing: 3) {
                Text("").frame(width: 78)
                ForEach(0..<24) { hour in
                    Text(hour % 3 == 0 ? String(format: "%02d", hour) : "")
                        .font(.omlxMono(9)).frame(maxWidth: .infinity)
                }
            }
            ScrollView {
                LazyVStack(spacing: 3) {
                    ForEach(days.reversed()) { day in
                        HStack(spacing: 3) {
                            Text(day.date).font(.omlxMono(10)).frame(width: 78, alignment: .leading)
                            ForEach(0..<24) { hour in
                                // The server always emits 24 cells; guard anyway so a
                                // short array can never index out of range.
                                let count = hour < day.tokens.count ? day.tokens[hour] : 0
                                RoundedRectangle(cornerRadius: 2)
                                    .fill(count == 0 ? Color.secondary.opacity(0.1) :
                                            Color.green.opacity(0.2 + 0.8 * sqrt(Double(count) / peak)))
                                    .frame(height: 14)
                                    .help(cellLabel(day.date, hour, count))
                                    .accessibilityLabel(cellLabel(day.date, hour, count))
                            }
                        }
                    }
                }
            }.frame(height: min(220, CGFloat(days.count * 17)))
        }.padding(.horizontal, 18)
    }

    private func cellLabel(_ date: String, _ hour: Int, _ count: Int) -> String {
        String(localized: "status.usage.heatmap.cell",
               defaultValue: "\(date) \(String(format: "%02d", hour)):00 · \(count.formatted()) tokens",
               comment: "Tooltip and accessibility label for one heatmap cell; placeholders are the date, the two-digit hour, and a formatted token count")
    }

    private func compact(_ count: Int) -> String {
        count.formatted(.number.notation(.compactName).precision(.fractionLength(0...1)))
    }

    private func speed(_ value: Double?) -> String {
        value.map { String(format: "%.1f", $0) } ?? "—"
    }

    private func detail(_ row: UsageHistoryDTO.UsageTotalsDTO) -> String {
        String(localized: "status.usage.row.detail",
               defaultValue: "Prompt \(compact(row.promptTokens)) · output \(compact(row.completionTokens)) · cached \(compact(row.cachedTokens))",
               comment: "Per-model breakdown sublabel; placeholders are compact prompt, output, and cached token counts")
    }
}
