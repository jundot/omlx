"""Custom AppKit widgets for oMLX app."""

from AppKit import (
    NSCommandKeyMask,
    NSPasteboard,
    NSPasteboardTypeString,
    NSSecureTextField,
)


class PastableSecureTextField(NSSecureTextField):
    """NSSecureTextField subclass that supports Cmd+V paste.

    PyObjC's NSSecureTextField does not handle paste from clipboard.
    This subclass intercepts Cmd+V and reads from the system pasteboard.
    """

    def performKeyEquivalent_(self, event):
        if event.modifierFlags() & NSCommandKeyMask:
            chars = event.charactersIgnoringModifiers()
            if chars == "v":
                pasteboard = NSPasteboard.generalPasteboard()
                string = pasteboard.stringForType_(NSPasteboardTypeString)
                if string:
                    self.setStringValue_(string)
                    return True
        return super().performKeyEquivalent_(event)
