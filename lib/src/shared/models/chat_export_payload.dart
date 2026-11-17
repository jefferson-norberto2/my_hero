/// Holds formatted text content and a sanitized file name for local export.
class ChatExportPayload {
  final String formattedContent;
  final String suggestedFileName;

  const ChatExportPayload({
    required this.formattedContent,
    required this.suggestedFileName,
  });

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is ChatExportPayload &&
        other.formattedContent == formattedContent &&
        other.suggestedFileName == suggestedFileName;
  }

  @override
  int get hashCode => Object.hash(formattedContent, suggestedFileName);

  @override
  String toString() =>
      'ChatExportPayload(suggestedFileName: $suggestedFileName, contentLength: ${formattedContent.length})';
}
