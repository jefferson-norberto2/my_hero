import 'dart:convert';
import 'dart:typed_data';

enum AttachmentType {
  image,
  audio,
  document,
}

class ChatAttachment {
  final String name;
  final AttachmentType type;
  final Uint8List bytes;
  final String? textContent;

  const ChatAttachment({
    required this.name,
    required this.type,
    required this.bytes,
    this.textContent,
  });

  String get sizeFormatted {
    if (bytes.isEmpty) return '0 B';
    final kb = bytes.length / 1024;
    if (kb >= 1024) {
      final mb = kb / 1024;
      return '${mb.toStringAsFixed(1)} MB';
    }
    return '${kb.toStringAsFixed(1)} KB';
  }

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'type': type.name,
      'bytes': base64Encode(bytes),
      'textContent': textContent,
    };
  }

  factory ChatAttachment.fromJson(Map<String, dynamic> json) {
    final typeStr = json['type'] as String? ?? 'document';
    final type = AttachmentType.values.firstWhere(
      (t) => t.name == typeStr,
      orElse: () => AttachmentType.document,
    );

    return ChatAttachment(
      name: json['name'] as String? ?? 'Anexo',
      type: type,
      bytes: base64Decode(json['bytes'] as String),
      textContent: json['textContent'] as String?,
    );
  }
}
