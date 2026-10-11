import 'package:flutter/material.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../../controller/chat_controller.dart';

class ChatDrawer extends StatelessWidget {
  final ChatController controller;

  const ChatDrawer({super.key, required this.controller});

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Row(
                children: [
                  CircleAvatar(
                    backgroundColor: Theme.of(context).colorScheme.primaryContainer,
                    child: const Icon(Icons.bolt, color: Colors.teal),
                  ),
                  const SizedBox(width: 12),
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'My Hero',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Text(
                          'Histórico de Conversas',
                          style: TextStyle(fontSize: 12, color: Colors.grey),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: FilledButton.icon(
                onPressed: () {
                  controller.createNewConversation();
                  Navigator.of(context).pop();
                },
                icon: const Icon(Icons.add),
                label: const Text('Nova Conversa'),
              ),
            ),
            const Divider(height: 24),
            Expanded(
              child: SignalBuilder(
                builder: (context) {
                  final convs = controller.conversations.value;
                  final currentId = controller.currentConversationId.value;

                  if (convs.isEmpty) {
                    return const Center(
                      child: Text(
                        'Nenhuma conversa ainda',
                        style: TextStyle(color: Colors.grey),
                      ),
                    );
                  }

                  return ListView.builder(
                    itemCount: convs.length,
                    itemBuilder: (context, index) {
                      final item = convs[index];
                      final isSelected = item.id == currentId;

                      return ListTile(
                        selected: isSelected,
                        selectedTileColor:
                            Theme.of(context).colorScheme.primaryContainer.withAlpha(80),
                        leading: Icon(
                          isSelected
                              ? Icons.chat_bubble
                              : Icons.chat_bubble_outline,
                          color: isSelected ? Colors.teal : Colors.grey[600],
                        ),
                        title: Text(
                          item.title,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            fontWeight:
                                isSelected ? FontWeight.bold : FontWeight.normal,
                          ),
                        ),
                        subtitle: Text(
                          '${item.is2B ? "Gemma 2B" : "Gemma 4B"} • ${item.messages.length} msgs',
                          style: const TextStyle(fontSize: 11),
                        ),
                        trailing: IconButton(
                          icon: const Icon(Icons.delete_outline, size: 20),
                          tooltip: 'Excluir conversa',
                          onPressed: () {
                            controller.deleteConversation(item.id);
                          },
                        ),
                        onTap: () {
                          controller.selectConversation(item.id);
                          Navigator.of(context).pop();
                        },
                      );
                    },
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
