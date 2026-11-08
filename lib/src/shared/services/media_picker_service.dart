import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import '../../features/chat/data/adapter/chat_attachment.dart';

class MediaPickerService {
  final ImagePicker _picker;

  MediaPickerService({ImagePicker? picker}) : _picker = picker ?? ImagePicker();

  Future<ChatAttachment?> pickImageFromGallery({int imageQuality = 80}) async {
    final XFile? file = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: imageQuality,
    );
    if (file == null) return null;
    final bytes = await file.readAsBytes();
    return ChatAttachment(
      name: file.name.isNotEmpty ? file.name : 'imagem.png',
      type: AttachmentType.image,
      bytes: bytes,
    );
  }

  Future<ChatAttachment?> pickImageFromCamera({int imageQuality = 80}) async {
    final XFile? file = await _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: imageQuality,
    );
    if (file == null) return null;
    final bytes = await file.readAsBytes();
    return ChatAttachment(
      name: file.name.isNotEmpty ? file.name : 'foto.png',
      type: AttachmentType.image,
      bytes: bytes,
    );
  }

  Future<ChatAttachment?> pickAudioFile() async {
    final result = await FilePicker.pickFiles(
      type: FileType.audio,
      withData: true,
    );
    if (result == null || result.files.isEmpty) return null;
    final file = result.files.single;

    Uint8List? bytes = file.bytes;
    if (bytes == null && file.path != null) {
      bytes = await File(file.path!).readAsBytes();
    }
    if (bytes == null) return null;

    return ChatAttachment(
      name: file.name,
      type: AttachmentType.audio,
      bytes: bytes,
    );
  }

  Future<ChatAttachment?> pickDocumentFile() async {
    final result = await FilePicker.pickFiles(
      type: FileType.any,
      withData: true,
    );
    if (result == null || result.files.isEmpty) return null;
    final file = result.files.single;

    Uint8List? bytes = file.bytes;
    if (bytes == null && file.path != null) {
      bytes = await File(file.path!).readAsBytes();
    }
    if (bytes == null) return null;

    String? textContent;
    try {
      textContent = utf8.decode(bytes);
    } catch (_) {
      // Arquivo binário não conversível para texto utf8 (ex: PDF ou binário)
    }

    return ChatAttachment(
      name: file.name,
      type: AttachmentType.document,
      bytes: bytes,
      textContent: textContent,
    );
  }
}
