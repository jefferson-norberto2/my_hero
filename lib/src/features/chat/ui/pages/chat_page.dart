import 'package:material_ui/material_ui.dart';
import 'package:get_it/get_it.dart';
import 'package:signals_flutter/signals_flutter.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_drawer.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_input_bar.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_message_list.dart';
import 'package:my_hero/src/features/chat/ui/widgets/model_selector_widget.dart';
import 'package:my_hero/src/features/chat/ui/widgets/model_switch_dialog.dart';

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
            final activeSpec = _controller.activeLoadedModelSpec;
            final maxTokens = _controller.maxTokens.value;
            final thinking =
                _controller.enableThinking.value && activeSpec.supportsThinking;
            final backendLabel = _controller.backendDisplaySummary;
            final styleLabel = _controller.responseStyle.value.label;

            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  conv?.title ?? 'Local Chat',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
                Text(
                  isLoaded
                      ? '${activeSpec.displayName} ($backendLabel) • ${AppConstants.formatTokenCount(maxTokens)}${thinking ? " • Thinking" : ""} • $styleLabel'
                      : 'Nenhum modelo carregado',
                  style: const TextStyle(fontSize: 11),
                  overflow: TextOverflow.ellipsis,
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
                tooltip: 'Configurações da conversa',
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
                final hasMessages = _controller.chatHistory.value.isNotEmpty;
                if (isLoaded) return const SizedBox.shrink();
                final maxHeight = hasMessages
                    ? MediaQuery.sizeOf(context).height * 0.62
                    : MediaQuery.sizeOf(context).height * 0.72;
                return ConstrainedBox(
                  constraints: BoxConstraints(maxHeight: maxHeight),
                  child: SingleChildScrollView(
                    child: Center(
                      child: ModelSelectorWidget(controller: _controller),
                    ),
                  ),
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
