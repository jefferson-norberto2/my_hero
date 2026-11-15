import 'package:my_hero/src/shared/models/chat_stream_chunk.dart';

/// Incrementally parses Ollama streaming chunks, separating reasoning from final answer content.
///
/// Supports both:
/// 1. Native Ollama `message.thinking` / `thinking` JSON fields.
/// 2. Inline `<think>...</think>` tags inside `message.content` / `response`.
class OllamaThinkingParser {
  static const String _openTag = '<think>';
  static const String _closeTag = '</think>';

  bool _insideThinkTag = false;
  bool _hasStartedContent = false;
  bool _nativeThinkingActive = false;
  bool _justClosedThinkTag = false;
  String _buffer = '';

  bool get isInsideThinkTag => _insideThinkTag;

  /// Parses an Ollama NDJSON chunk from either `/api/chat` or `/api/generate`.
  ChatStreamChunk parseJsonChunk(Map<String, dynamic> json) {
    final message = json['message'] as Map<String, dynamic>?;
    final nativeThinking =
        (message?['thinking'] as String?) ?? (json['thinking'] as String?) ?? '';
    final rawContent =
        (message?['content'] as String?) ?? (json['response'] as String?) ?? '';
    final isDone = json['done'] == true;

    return feed(
      nativeThinking: nativeThinking,
      rawContent: rawContent,
      isDone: isDone,
    );
  }

  /// Processes incoming thinking/content fragments and returns the extracted deltas.
  ChatStreamChunk feed({
    String nativeThinking = '',
    String rawContent = '',
    bool isDone = false,
  }) {
    final thinkingOut = StringBuffer();
    final contentOut = StringBuffer();

    if (nativeThinking.isNotEmpty) {
      thinkingOut.write(nativeThinking);
      _nativeThinkingActive = true;
    }

    if (rawContent.isNotEmpty) {
      _buffer += rawContent;
    }

    while (_buffer.isNotEmpty) {
      if (_insideThinkTag) {
        final closeIdx = _buffer.indexOf(_closeTag);
        if (closeIdx != -1) {
          thinkingOut.write(_buffer.substring(0, closeIdx));
          _buffer = _buffer.substring(closeIdx + _closeTag.length);
          _insideThinkTag = false;
          _justClosedThinkTag = true;
          continue;
        }

        if (isDone) {
          thinkingOut.write(_buffer);
          _buffer = '';
          _insideThinkTag = false;
          break;
        }

        final holdLen = _longestSuffixPrefixLength(_buffer, _closeTag);
        final safeEnd = _buffer.length - holdLen;
        if (safeEnd > 0) {
          thinkingOut.write(_buffer.substring(0, safeEnd));
          _buffer = _buffer.substring(safeEnd);
        }
        break;
      } else {
        if (_justClosedThinkTag) {
          _buffer = _buffer.replaceFirst(RegExp(r'^\s+'), '');
          if (_buffer.isEmpty) {
            break;
          }
          _justClosedThinkTag = false;
        }

        // Hold leading whitespace before the first token in case `<think>` follows immediately.
        if (!_hasStartedContent && _buffer.trim().isEmpty && !isDone) {
          break;
        }

        final openIdx = _buffer.indexOf(_openTag);
        if (openIdx != -1) {
          final beforeTag = _buffer.substring(0, openIdx);
          if (beforeTag.isNotEmpty &&
              (_hasStartedContent || beforeTag.trim().isNotEmpty)) {
            contentOut.write(beforeTag);
            _hasStartedContent = true;
            _nativeThinkingActive = false;
          }
          _buffer = _buffer.substring(openIdx + _openTag.length);
          _insideThinkTag = true;
          continue;
        }

        if (isDone) {
          if (_buffer.isNotEmpty) {
            contentOut.write(_buffer);
            _buffer = '';
            _hasStartedContent = true;
            _nativeThinkingActive = false;
          }
          break;
        }

        final holdLen = _longestSuffixPrefixLength(_buffer, _openTag);
        final safeEnd = _buffer.length - holdLen;
        if (safeEnd > 0) {
          final chunk = _buffer.substring(0, safeEnd);
          contentOut.write(chunk);
          _buffer = _buffer.substring(safeEnd);
          if (chunk.trim().isNotEmpty || _hasStartedContent) {
            _hasStartedContent = true;
            _nativeThinkingActive = false;
          }
        }
        break;
      }
    }

    if (contentOut.isNotEmpty) {
      _hasStartedContent = true;
      _nativeThinkingActive = false;
    }

    final bool currentlyThinking;
    if (isDone) {
      _insideThinkTag = false;
      _nativeThinkingActive = false;
      currentlyThinking = false;
    } else if (_insideThinkTag) {
      currentlyThinking = true;
    } else if (nativeThinking.isNotEmpty && contentOut.isEmpty) {
      currentlyThinking = true;
    } else if (_nativeThinkingActive && !_hasStartedContent) {
      currentlyThinking = true;
    } else {
      currentlyThinking = false;
    }

    return ChatStreamChunk(
      thinkingDelta: thinkingOut.toString(),
      contentDelta: contentOut.toString(),
      isThinking: currentlyThinking,
      isDone: isDone,
    );
  }

  /// Flushes any remaining buffered characters at stream completion.
  ChatStreamChunk flush() {
    return feed(isDone: true);
  }

  /// Resets the parser state for a new response stream.
  void reset() {
    _insideThinkTag = false;
    _hasStartedContent = false;
    _nativeThinkingActive = false;
    _justClosedThinkTag = false;
    _buffer = '';
  }

  /// Returns the length of the longest suffix of [text] that matches a prefix of [tag].
  int _longestSuffixPrefixLength(String text, String tag) {
    final maxCheck = text.length < tag.length - 1 ? text.length : tag.length - 1;
    for (int len = maxCheck; len >= 1; len--) {
      final suffix = text.substring(text.length - len);
      if (tag.startsWith(suffix)) {
        return len;
      }
    }
    return 0;
  }
}
