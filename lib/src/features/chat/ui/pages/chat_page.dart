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
            SnackBar(
              content: Text(error),
              backgroundColor: Theme.of(context).colorScheme.error,
              behavior: SnackBarBehavior.floating,
            ),
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
            final isConnected = _controller.isOllamaConnected.value;
            final modelName = _controller.selectedModelName.value;

            String subtitle;
            if (!isConnected) {
              subtitle = 'Ollama desconectado (localhost:11434)';
            } else if (isLoaded && modelName.isNotEmpty) {
              subtitle = '$modelName • ${_controller.responseStyle.value.label}';
            } else if (modelName.isNotEmpty) {
              subtitle = '$modelName (não carregado na memória)';
            } else {
              subtitle = 'Nenhum modelo selecionado';
            }

            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  conv?.title ?? 'Ollama Chat',
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  overflow: TextOverflow.ellipsis,
                ),
                Text(
                  subtitle,
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
              final isGenerating = _controller.isGenerating.value;

              return IconButton(
                icon: const Icon(Icons.tune),
                tooltip: 'Trocar modelo / Estilo',
                onPressed: isGenerating
                    ? null
                    : () => ModelSwitchDialog.show(context, _controller),
              );
            },
          ),
          SignalBuilder(
            builder: (context) {
              final isGenerating = _controller.isGenerating.value;

              return IconButton(
                icon: const Icon(Icons.refresh),
                tooltip: 'Atualizar conexão com Ollama',
                onPressed: isGenerating ? null : _controller.refreshModels,
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
                return Padding(
                  padding: const EdgeInsets.only(bottom: 12.0),
                  child: ModelSelectorWidget(controller: _controller),
                );
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
