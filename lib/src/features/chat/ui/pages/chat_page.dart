import 'package:flutter/material.dart';
import 'package:get_it/get_it.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../../controller/chat_controller.dart';
import '../widgets/chat_input_bar.dart';
import '../widgets/chat_message_list.dart';
import '../widgets/model_selector_widget.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  late final ChatController _controller;

  @override
  void initState() {
    super.initState();
    _controller = GetIt.I<ChatController>();

    // Efeito para exibir mensagens de erro quando surgirem
    effect(() {
      final error = _controller.errorMessage.value;
      if (error != null && mounted) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(error)),
          );
        });
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Gemma 4 Local'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          SignalBuilder(
            builder: (context) {
              final isLoaded = _controller.isModelLoaded.value;
              final isGenerating = _controller.isGenerating.value;

              if (!isLoaded) return const SizedBox.shrink();

              return IconButton(
                icon: const Icon(Icons.refresh),
                tooltip: 'Reiniciar modelo',
                onPressed: isGenerating ? null : _controller.reset,
              );
            },
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            SignalBuilder(
              builder: (context) {
                final isLoaded = _controller.isModelLoaded.value;
                if (isLoaded) return const SizedBox.shrink();
                return ModelSelectorWidget(controller: _controller);
              },
            ),
            Expanded(
              child: ChatMessageList(controller: _controller),
            ),
            SignalBuilder(
              builder: (context) {
                if (_controller.isGenerating.value) {
                  return const Padding(
                    padding: EdgeInsets.symmetric(vertical: 8.0),
                    child: LinearProgressIndicator(),
                  );
                }
                return const SizedBox.shrink();
              },
            ),
            ChatInputBar(controller: _controller),
          ],
        ),
      ),
    );
  }
}
