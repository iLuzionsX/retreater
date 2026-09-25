import SwiftUI

struct SettingsView: View {
    @AppStorage("LiveTR3.startsRuntimeAutomatically") private var startsRuntimeAutomatically = true
    @AppStorage("LiveTR3.showsAdvancedRuntimeDetails") private var showsAdvancedRuntimeDetails = false
    @AppStorage("LiveTR3.glossary") private var glossary = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Settings")
                    .font(.title2.weight(.semibold))
                Text("Configure how the local caption engine starts and how much system detail LiveTR3 shows.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Form {
                Section {
                    Toggle(isOn: $startsRuntimeAutomatically) {
                        SettingLabel(
                            title: "Start Local Engine",
                            detail: "Launch the local caption engine when LiveTR3 opens."
                        )
                    }
                } header: {
                    Text("Local Engine")
                } footer: {
                    Text("Turn this off when you want to open the Mac app without starting capture services.")
                }

                Section {
                    TextField("Names and terms", text: $glossary, axis: .vertical)
                        .lineLimit(2...4)
                        .accessibilityLabel("Glossary")
                        .accessibilityHint("Comma-separated names and terms to prefer in captions.")
                } header: {
                    Text("Glossary")
                } footer: {
                    Text("Optional names and terms, separated by commas. LiveTR3 adds them to the translation prompt for finalized captions.")
                }

                Section {
                    Toggle(isOn: $showsAdvancedRuntimeDetails) {
                        SettingLabel(
                            title: "Show Advanced Engine Details",
                            detail: "Reveal local IPC and diagnostic information in the Local Engine view."
                        )
                    }
                } header: {
                    Text("Diagnostics")
                } footer: {
                    Text("Advanced details are informational. They do not change capture, transcript, or projector behavior.")
                }
            }
            .formStyle(.grouped)
        }
        .padding(24)
        .frame(width: 520)
    }
}

private struct SettingLabel: View {
    let title: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
            Text(detail)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
