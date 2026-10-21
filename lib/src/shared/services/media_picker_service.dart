import 'dart:typed_data';
import 'package:image_picker/image_picker.dart';

class MediaPickerService {
  final ImagePicker _picker;

  MediaPickerService({ImagePicker? picker}) : _picker = picker ?? ImagePicker();

  Future<Uint8List?> pickImageFromGallery({int imageQuality = 80}) async {
    return pickImage(source: ImageSource.gallery, imageQuality: imageQuality);
  }

  Future<Uint8List?> pickImageFromCamera({int imageQuality = 80}) async {
    return pickImage(source: ImageSource.camera, imageQuality: imageQuality);
  }

  Future<Uint8List?> pickImage({
    ImageSource source = ImageSource.gallery,
    int imageQuality = 80,
  }) async {
    final XFile? file = await _picker.pickImage(
      source: source,
      imageQuality: imageQuality,
    );
    if (file == null) return null;
    return await file.readAsBytes();
  }
}
