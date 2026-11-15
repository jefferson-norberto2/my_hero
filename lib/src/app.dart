import 'package:flutter/material.dart';
import 'package:my_hero/src/core/routes/app_routes.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';

/// Root Material application configured with GoRouter.
class GemmaChatApp extends StatelessWidget {
  const GemmaChatApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'Gemma Local Chat',
      theme: AppTheme.lightTheme,
      routerConfig: AppRoutes.router,
    );
  }
}
