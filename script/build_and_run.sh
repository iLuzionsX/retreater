#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="LiveTR3"
PACKAGE_DIR="$ROOT/macos/LiveTR3Mac"
BUNDLE="$ROOT/dist/$APP_NAME.app"
APP_ICON="$PACKAGE_DIR/Resources/AppIcon.icns"

verify=false
if [[ "${1:-}" == "--verify" ]]; then
  verify=true
fi

pkill -x "$APP_NAME" >/dev/null 2>&1 || true

cd "$PACKAGE_DIR"
# Command Line Tools ship Swift but not the SwiftUI macro plugin. Prefer a
# local Xcode/Xcode-beta plugin directory when the CLT toolchain cannot see it.
SWIFT_BUILD_ARGS=(-c debug)
PLUGIN_DIR=""
for candidate in \
  "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/usr/lib/swift/host/plugins" \
  "/Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/usr/lib/swift/host/plugins" \
  "/Library/Developer/CommandLineTools/usr/lib/swift/host/plugins"
do
  if [[ -f "$candidate/libSwiftUIMacros.dylib" ]]; then
    PLUGIN_DIR="$candidate"
    break
  fi
done
if [[ -n "$PLUGIN_DIR" && ! -f "/Library/Developer/CommandLineTools/usr/lib/swift/host/plugins/libSwiftUIMacros.dylib" ]]; then
  SWIFT_BUILD_ARGS+=(-Xswiftc -plugin-path -Xswiftc "$PLUGIN_DIR")
fi
swift build "${SWIFT_BUILD_ARGS[@]}"

EXECUTABLE="$(swift build "${SWIFT_BUILD_ARGS[@]}" --show-bin-path)/$APP_NAME"

rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"
cp "$EXECUTABLE" "$BUNDLE/Contents/MacOS/$APP_NAME"
cp "$APP_ICON" "$BUNDLE/Contents/Resources/AppIcon.icns"
cat > "$BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>$APP_NAME</string>
  <key>CFBundleIdentifier</key>
  <string>com.livetr3.mac</string>
  <key>CFBundleName</key>
  <string>$APP_NAME</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>14.0</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>LiveTR3 captures microphone audio for local transcription and translation.</string>
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
</dict>
</plist>
PLIST

"$ROOT/script/package_engine.sh" "$BUNDLE"

if [[ "${SKIP_OPEN:-}" != "1" ]]; then
  /usr/bin/open -n "$BUNDLE"
fi

if [[ "$verify" == true ]]; then
  sleep 2
  pgrep -x "$APP_NAME" >/dev/null
  echo "$APP_NAME is running from $BUNDLE"
fi
