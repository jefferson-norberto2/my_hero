import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter_gemma/flutter_gemma.dart';

class ChatMessage {
  final String text;
  final bool isUser;
  final Uint8List? imageBytes;

  ChatMessage({
    required this.text,
    required this.isUser,
    this.imageBytes,
  });

  ChatMessage copyWith({
    String? text,
    bool? isUser,
    Uint8List? imageBytes,
  }) {
    return ChatMessage(
      text: text ?? this.text,
      isUser: isUser ?? this.isUser,
      imageBytes: imageBytes ?? this.imageBytes,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'text': text,
      'isUser': isUser,
      'imageBytes': imageBytes != null ? base64Encode(imageBytes!) : null,
    };
  }

  factory ChatMessage.fromJson(Map<String, dynamic> json) {
    return ChatMessage(
      text: json['text'] as String? ?? '',
      isUser: json['isUser'] as bool? ?? false,
      imageBytes: json['imageBytes'] != null
          ? base64Decode(json['imageBytes'] as String)
          : null,
    );
  }
}

class ChatMessageAdapter {
  static Message toGemmaMessage(ChatMessage message) {
    if (message.imageBytes != null) {
      return Message(
        text: message.text,
        imageBytes: message.imageBytes!,
        isUser: message.isUser,
      );
    }
    return Message.text(text: message.text, isUser: message.isUser);
  }
}
