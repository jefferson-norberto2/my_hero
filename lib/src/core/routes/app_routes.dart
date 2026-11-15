import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:my_hero/src/features/chat/ui/pages/chat_page.dart';
import 'package:my_hero/src/features/settings/data/models/settings_args.dart';
import 'package:my_hero/src/features/settings/ui/pages/settings_screen.dart';

/// Application route constants and GoRouter configuration.
class AppRoutes {
  static const String initial = '/';
  static const String chat = '/chat';
  static const String settings = '/settings';

  static final GoRouter router = createRouter();

  /// Creates a [GoRouter] instance starting at [initialLocation].
  static GoRouter createRouter({String initialLocation = initial}) {
    return GoRouter(
      initialLocation: initialLocation,
      routes: [
        GoRoute(
          path: initial,
          builder: (context, state) => const ChatPage(),
        ),
        GoRoute(
          path: chat,
          builder: (context, state) => const ChatPage(),
        ),
        GoRoute(
          path: settings,
          builder: (context, state) {
            final args =
                state.extra is SettingsArgs ? state.extra as SettingsArgs : null;
            return SettingsScreen(initialArgs: args);
          },
        ),
      ],
      errorBuilder: (context, state) => Scaffold(
        body: Center(
          child: Text('Rota não encontrada: ${state.uri}'),
        ),
      ),
    );
  }
}
