import 'package:material_ui/material_ui.dart';
import 'package:my_hero/src/features/chat/ui/pages/chat_page.dart';

class AppRoutes {
  static const String initial = '/';
  static const String chat = '/chat';

  static Route<dynamic> onGenerateRoute(RouteSettings settings) {
    switch (settings.name) {
      case initial:
      case chat:
        return MaterialPageRoute(builder: (_) => const ChatPage());
      default:
        return MaterialPageRoute(
          builder: (_) => Scaffold(
            body: Center(
              child: Text('Rota não encontrada: ${settings.name}'),
            ),
          ),
        );
    }
  }
}
