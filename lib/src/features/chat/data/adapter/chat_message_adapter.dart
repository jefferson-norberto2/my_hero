import 'dart:convert';
import 'dart:typed_data';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';

/// Represents a chat message exchanged between the user and the assistant.
class ChatMessage {
  final String content;
  final String thinkingContent;
  final bool isThinking;
  final bool isUser;
  final Uint8List? imageBytes;
  final ChatAttachment? attachment;

  /// Backward-compatible alias for [content].
  String get text => content;

  /// Whether this message has non-empty reasoning content or is actively thinking.
  bool get hasThinking => isThinking || thinkingContent.trim().isNotEmpty;

  ChatMessage({
    String? text,
    String? content,
    this.thinkingContent = '',
    this.isThinking = false,
    required this.isUser,
    this.imageBytes,
    this.attachment,
  }) : content = content ?? text ?? '';

  ChatMessage copyWith({
    String? text,
    String? content,
    String? thinkingContent,
    bool? isThinking,
    bool? isUser,
    Uint8List? imageBytes,
    ChatAttachment? attachment,
  }) {
    return ChatMessage(
      content: content ?? text ?? this.content,
      thinkingContent: thinkingContent ?? this.thinkingContent,
      isThinking: isThinking ?? this.isThinking,
      isUser: isUser ?? this.isUser,
      imageBytes: imageBytes ?? this.imageBytes,
      attachment: attachment ?? this.attachment,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'text': content,
      'content': content,
      'thinkingContent': thinkingContent,
      'isThinking': isThinking,
      'isUser': isUser,
      'imageBytes': imageBytes != null ? base64Encode(imageBytes!) : null,
      'attachment': attachment?.toJson(),
    };
  }

  factory ChatMessage.fromJson(Map<String, dynamic> json) {
    ChatAttachment? attachment;
    if (json['attachment'] != null) {
      attachment = ChatAttachment.fromJson(
        json['attachment'] as Map<String, dynamic>,
      );
    }

    final rawImage = json['imageBytes'] != null
        ? base64Decode(json['imageBytes'] as String)
        : null;

    final effectiveImage = rawImage ??
        (attachment != null && attachment.type == AttachmentType.image
            ? attachment.bytes
            : null);

    final parsedContent =
        json['content'] as String? ?? json['text'] as String? ?? '';

    return ChatMessage(
      content: parsedContent,
      thinkingContent: json['thinkingContent'] as String? ?? '',
      isThinking: json['isThinking'] as bool? ?? false,
      isUser: json['isUser'] as bool? ?? false,
      imageBytes: effectiveImage,
      attachment: attachment,
    );
  }
}

/// Converts [ChatMessage] instances into Ollama API message payloads.
class ChatMessageAdapter {
  static Map<String, dynamic> toOllamaMessage(ChatMessage message) {
    String content = message.content;
    final att = message.attachment;

    if (att != null) {
      if (att.type == AttachmentType.document &&
          att.textContent != null &&
          att.textContent!.isNotEmpty) {
        content =
            '--- Arquivo anexado: ${att.name} ---\n${att.textContent}\n--- Fim do arquivo ---\n\n$content';
      } else if (att.type == AttachmentType.document) {
        content =
            '[Documento anexado: ${att.name} (${att.sizeFormatted})]\n\n$content';
      } else if (att.type == AttachmentType.audio) {
        content =
            '[Áudio anexado: ${att.name} (${att.sizeFormatted})]\n\n$content';
      }
    }

    final map = <String, dynamic>{
      'role': message.isUser ? 'user' : 'assistant',
      'content': content,
    };

    final effectiveImageBytes = message.imageBytes ??
        (att != null && att.type == AttachmentType.image ? att.bytes : null);

    if (effectiveImageBytes != null) {
      map['images'] = [base64Encode(effectiveImageBytes)];
    }

    return map;
  }
}
