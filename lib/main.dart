import 'package:flutter/material.dart';
import 'src/app.dart';
import 'src/core/di/injection.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await setupDependencies();
  runApp(const GemmaChatApp());
}
