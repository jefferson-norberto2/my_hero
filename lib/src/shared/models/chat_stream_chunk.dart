/// Represents a parsed incremental chunk emitted from the model stream.
class ChatStreamChunk {
  final String thinkingDelta;
  final String contentDelta;
  final bool isThinking;
  final bool isDone;

  const ChatStreamChunk({
    this.thinkingDelta = '',
    this.contentDelta = '',
    this.isThinking = false,
    this.isDone = false,
  });

  bool get isEmpty =>
      thinkingDelta.isEmpty && contentDelta.isEmpty && !isThinking && !isDone;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is ChatStreamChunk &&
        other.thinkingDelta == thinkingDelta &&
        other.contentDelta == contentDelta &&
        other.isThinking == isThinking &&
        other.isDone == isDone;
  }

  @override
  int get hashCode =>
      Object.hash(thinkingDelta, contentDelta, isThinking, isDone);

  @override
  String toString() =>
      'ChatStreamChunk(thinkingDelta: $thinkingDelta, contentDelta: $contentDelta, isThinking: $isThinking, isDone: $isDone)';
}
