import 'dart:io';
import 'dart:typed_data';
import 'package:image_picker/image_picker.dart';

class MediaPickerService {
  final ImagePicker _picker;

  MediaPickerService({ImagePicker? picker}) : _picker = picker ?? ImagePicker();

  Future<Uint8List?> pickImageFromGallery({int imageQuality = 80}) async {
    final XFile? file = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: imageQuality,
    );
    if (file == null) return null;
    return await File(file.path).readAsBytes();
  }
}
