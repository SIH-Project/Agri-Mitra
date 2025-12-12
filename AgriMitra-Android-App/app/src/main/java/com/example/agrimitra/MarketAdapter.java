package com.example.agrimitra;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import java.util.List;

public class MarketAdapter extends RecyclerView.Adapter<MarketAdapter.ViewHolder> {
    private List<MarketPrice> marketList;
    private TranslationHelper translationHelper;

    public MarketAdapter(Context context, List<MarketPrice> marketList) {
        this.marketList = marketList;
        this.translationHelper = new TranslationHelper(context);
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_market_price, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        MarketPrice price = marketList.get(position);
        
        // Initialize translation and translate labels
        translationHelper.initializeTranslation(() -> {
            translationHelper.translateTextView(holder.commodity, price.getCommodity());
            translationHelper.translateText("City: " + price.getCity(), translatedText -> {
                holder.city.setText(translatedText);
            });
            translationHelper.translateText("Min: " + price.getMinPrice(), translatedText -> {
                holder.min.setText(translatedText);
            });
            translationHelper.translateText("Max: " + price.getMaxPrice(), translatedText -> {
                holder.max.setText(translatedText);
            });
            translationHelper.translateText("Model: " + price.getModelPrice(), translatedText -> {
                holder.model.setText(translatedText);
            });
            translationHelper.translateText("Date: " + price.getDate(), translatedText -> {
                holder.date.setText(translatedText);
            });
        });
    }

    @Override
    public int getItemCount() {
        return marketList.size();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        TextView commodity, city, min, max, model, date;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            commodity = itemView.findViewById(R.id.commodity);
            city = itemView.findViewById(R.id.city);
            min = itemView.findViewById(R.id.minPrice);
            max = itemView.findViewById(R.id.maxPrice);
            model = itemView.findViewById(R.id.modelPrice);
            date = itemView.findViewById(R.id.date);
        }
    }
}
