import SwiftUI

struct ProgressBar: View {
    let progress: Double
    @State var colors: [Color]? = nil
    @Environment(\.omlxTheme) private var theme

    init(progress: Double) {
        self.progress = progress
    }

    init(progress: Double, tint: Color? = nil) {
        self.progress = progress
        if let color = tint {
            self._colors = State(initialValue: [color])
        }
    }

    init(progress: Double, colors: [Color]? = nil) {
        self.progress = progress
        self._colors = State(initialValue: colors)
    }

    var body: some View {
        ProgressView(value: progress)
            .progressViewStyle(OMLXLinearProgressViewStyle(colors: colors))
    }
}

struct OMLXLinearProgressViewStyle: ProgressViewStyle {
    @Environment(\.omlxTheme) private var theme
    var colors: [Color]? = nil
    func makeBody(configuration: Configuration) -> some View {
        let progress = configuration.fractionCompleted ?? 0
        let progressShapeStyle: AnyShapeStyle = {
            if let colors = colors {
                return AnyShapeStyle(LinearGradient(
                    colors: colors,
                    startPoint: .leading,
                    endPoint: .trailing
                ))
            } else {
                return AnyShapeStyle(.tint)
            }
        }()
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(theme.codeBg)
                Capsule()
                    .fill(progressShapeStyle)
                    .frame(width: geo.size.width * max(0, min(progress, 1)))
                    .animation(.easeOut(duration: 0.4), value: progress)
            }
        }
        .frame(height: 4)
    }
}

#Preview {
    @Previewable @State var progress = 0.3
    let downloadsColors = [Color(rgb24: 0x0A84FF), Color(rgb24: 0x5E5CE6)]
    let quantizationColors = [Color(rgb24: 0xFF2D55), Color(rgb24: 0xAF52DE)]

    VStack(spacing: 20) {
        ProgressBar(progress: progress)
        ProgressBar(progress: progress, tint: .red)
        ProgressBar(progress: progress, colors: downloadsColors)
        ProgressBar(progress: progress, colors: quantizationColors)

        Slider(value: $progress, in: 0.0...1.0)
    }
    .padding()
}
