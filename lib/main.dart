import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'app.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  FlutterGemma.initialize();
  runApp(const GemmaChatApp());
}
