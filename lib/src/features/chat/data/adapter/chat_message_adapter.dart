import 'dart:convert';
import 'dart:typed_data';
import 'chat_attachment.dart';

class ChatMessage {
  final String text;
  final bool isUser;
  final Uint8List? imageBytes;
  final ChatAttachment? attachment;

  ChatMessage({
    required this.text,
    required this.isUser,
    this.imageBytes,
    this.attachment,
  });

  ChatMessage copyWith({
    String? text,
    bool? isUser,
    Uint8List? imageBytes,
    ChatAttachment? attachment,
  }) {
    return ChatMessage(
      text: text ?? this.text,
      isUser: isUser ?? this.isUser,
      imageBytes: imageBytes ?? this.imageBytes,
      attachment: attachment ?? this.attachment,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'text': text,
      'isUser': isUser,
      'imageBytes': imageBytes != null ? base64Encode(imageBytes!) : null,
      'attachment': attachment?.toJson(),
    };
  }

  factory ChatMessage.fromJson(Map<String, dynamic> json) {
    ChatAttachment? attachment;
    if (json['attachment'] != null) {
      attachment = ChatAttachment.fromJson(json['attachment'] as Map<String, dynamic>);
    }

    final rawImage = json['imageBytes'] != null
        ? base64Decode(json['imageBytes'] as String)
        : null;

    final effectiveImage = rawImage ??
        (attachment != null && attachment.type == AttachmentType.image
            ? attachment.bytes
            : null);

    return ChatMessage(
      text: json['text'] as String? ?? '',
      isUser: json['isUser'] as bool? ?? false,
      imageBytes: effectiveImage,
      attachment: attachment,
    );
  }
}

class ChatMessageAdapter {
  static Map<String, dynamic> toOllamaMessage(ChatMessage message) {
    String content = message.text;
    final att = message.attachment;

    if (att != null) {
      if (att.type == AttachmentType.document &&
          att.textContent != null &&
          att.textContent!.isNotEmpty) {
        content =
            '--- Arquivo anexado: ${att.name} ---\n${att.textContent}\n--- Fim do arquivo ---\n\n$content';
      } else if (att.type == AttachmentType.document) {
        content = '[Documento anexado: ${att.name} (${att.sizeFormatted})]\n\n$content';
      } else if (att.type == AttachmentType.audio) {
        content = '[Áudio anexado: ${att.name} (${att.sizeFormatted})]\n\n$content';
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
