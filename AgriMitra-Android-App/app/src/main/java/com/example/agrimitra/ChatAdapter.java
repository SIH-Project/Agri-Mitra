package com.example.agrimitra;



import android.view.LayoutInflater;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.cardview.widget.CardView;
import androidx.recyclerview.widget.RecyclerView;

import java.lang.reflect.Array;
import java.util.ArrayList;

public class ChatAdapter extends RecyclerView.Adapter<ChatAdapter.ViewHolder> {

    ArrayList<String> sender, message;
    private TranslationHelper translationHelper;

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        CardView cv = (CardView) LayoutInflater.from(parent.getContext()).inflate(R.layout.chat_cardview, parent, false);


        return new ViewHolder(cv);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        CardView cv = holder.cardView;
        TextView title = cv.findViewById(R.id.sender_name);
        TextView mes = cv.findViewById(R.id.message_content);
        
        if (translationHelper == null) {
            translationHelper = new TranslationHelper(cv.getContext());
        }
        
        translationHelper.initializeTranslation(() -> {
            translationHelper.translateTextView(title, sender.get(position));
            translationHelper.translateTextView(mes, message.get(position));
        });
    }

    @Override
    public int getItemCount() {
        return message.size();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {

        CardView cardView;

        public ViewHolder(@NonNull CardView cv) {
            super(cv);
            this.cardView = cv;
        }
    }

    public ChatAdapter(ArrayList<String> s, ArrayList<String> m) {
        this.sender = s;
        this.message = m;
    }
}


