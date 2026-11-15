import 'package:flutter/material.dart';
import 'package:my_hero/src/app.dart';
import 'package:my_hero/src/core/di/injection.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await setupDependencies();
  runApp(const GemmaChatApp());
}
