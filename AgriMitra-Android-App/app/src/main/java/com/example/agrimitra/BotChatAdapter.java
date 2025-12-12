package com.example.agrimitra;



import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class BotChatAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {

    private List<ChatMessage> messages;
    private TranslationHelper translationHelper;
    private static final int USER = 0;
    private static final int BOT = 1;

    public BotChatAdapter(android.content.Context context, List<ChatMessage> messages) {
        this.messages = messages;
        this.translationHelper = new TranslationHelper(context);
    }

    @Override
    public int getItemViewType(int position) {
        return messages.get(position).isUser() ? USER : BOT;
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        if (viewType == USER) {
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_user_message, parent, false);
            return new UserViewHolder(view);
        } else {
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_bot_message, parent, false);
            return new BotViewHolder(view);
        }
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        ChatMessage msg = messages.get(position);
        translationHelper.initializeTranslation(() -> {
            if (holder instanceof UserViewHolder) {
                translationHelper.translateTextView(((UserViewHolder) holder).tvMessage, msg.getMessage());
            } else if (holder instanceof BotViewHolder) {
                translationHelper.translateTextView(((BotViewHolder) holder).tvMessage, msg.getMessage());
            }
        });
    }

    @Override
    public int getItemCount() {
        return messages.size();
    }

    static class UserViewHolder extends RecyclerView.ViewHolder {
        TextView tvMessage;
        UserViewHolder(View itemView) {
            super(itemView);
            tvMessage = itemView.findViewById(R.id.tvUserMessage);
        }
    }

    static class BotViewHolder extends RecyclerView.ViewHolder {
        TextView tvMessage;
        BotViewHolder(View itemView) {
            super(itemView);
            tvMessage = itemView.findViewById(R.id.tvBotMessage);
        }
    }
}
