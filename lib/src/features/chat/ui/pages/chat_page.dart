import 'package:flutter/material.dart';
import 'package:get_it/get_it.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../../controller/chat_controller.dart';
import '../widgets/chat_drawer.dart';
import '../widgets/chat_input_bar.dart';
import '../widgets/chat_message_list.dart';
import '../widgets/model_selector_widget.dart';
import '../widgets/model_switch_dialog.dart';

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
    _controller.init();

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
      drawer: ChatDrawer(controller: _controller),
      appBar: AppBar(
        title: SignalBuilder(
          builder: (context) {
            final conv = _controller.currentConversation;
            final isLoaded = _controller.isModelLoaded.value;
            final is2B = _controller.is2BModel.value;
            final isCpu = _controller.isCpu;

            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  conv?.title ?? 'My Hero',
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  overflow: TextOverflow.ellipsis,
                ),
                Text(
                  isLoaded
                      ? '${is2B ? "Gemma 2B" : "Gemma 4B"} (${isCpu ? "CPU" : "GPU"}) • ${_controller.responseStyle.value.label}'
                      : 'Nenhum modelo carregado',
                  style: const TextStyle(fontSize: 11),
                ),
              ],
            );
          },
        ),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          SignalBuilder(
            builder: (context) {
              final isLoaded = _controller.isModelLoaded.value;
              final isGenerating = _controller.isGenerating.value;

              if (!isLoaded) return const SizedBox.shrink();

              return IconButton(
                icon: const Icon(Icons.tune),
                tooltip: 'Alterar modelo (mantém contexto)',
                onPressed: isGenerating
                    ? null
                    : () => ModelSwitchDialog.show(context, _controller),
              );
            },
          ),
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
