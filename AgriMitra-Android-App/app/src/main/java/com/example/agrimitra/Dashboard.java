package com.example.agrimitra;

import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.widget.ImageButton;
import android.speech.tts.TextToSpeech;
import java.util.Locale;

import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.github.mikephil.charting.charts.BarChart;
import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.charts.PieChart;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.data.PieData;
import com.github.mikephil.charting.data.PieDataSet;
import com.github.mikephil.charting.data.PieEntry;
import com.github.mikephil.charting.formatter.ValueFormatter;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.util.ArrayList;
import java.util.Locale;

public class Dashboard extends AppCompatActivity {

    BarChart barChart_crop, barChart_rice, barChart_sugarcane, barChart_cotton, barChart_potato, barChart_yield_compare;
    LineChart lineChart_rainfall, lineChart_profit;
    PieChart pieChart_expenditure;

    TranslationHelper translationHelper;

    TextToSpeech tts;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_dashboard);


        translationHelper=new TranslationHelper(this);

        tts = new TextToSpeech(this, status -> {
            if(status != TextToSpeech.ERROR){
                tts.setLanguage(Locale.getDefault());
            }
        });

        // Initialize charts
        barChart_crop = findViewById(R.id.barChart_crop);
        barChart_rice = findViewById(R.id.barChart_rice);
        barChart_sugarcane = findViewById(R.id.barChart_sugarcane);
        barChart_cotton = findViewById(R.id.barChart_cotton);
        barChart_potato = findViewById(R.id.barChart_potato);
        barChart_yield_compare = findViewById(R.id.barChart_yield_compare);

        lineChart_rainfall = findViewById(R.id.lineChart_rainfall);
        lineChart_profit = findViewById(R.id.lineChart_profit);

        pieChart_expenditure = findViewById(R.id.pieChart_expenditure);

        // Setup charts with data
        setupCropBarChart(barChart_crop, new float[]{1.43f, 1.39f, 1.41f, 1.36f, 1.41f});
        setupCropBarChart(barChart_rice, new float[]{1.49f, 1.57f, 1.56f, 1.58f, 1.59f});
        setupCropBarChart(barChart_sugarcane, new float[]{30.8f, 32f, 32.8f, 34f, 34.4f});
        setupCropBarChart(barChart_cotton, new float[]{0.205f, 0.213f, 0.211f, 0.212f, 0.231f});
        setupCropBarChart(barChart_potato, new float[]{9.23f, 9.91f, 8.56f, 10.2f, 10.8f});

        setupLineChart(lineChart_rainfall, new float[]{800,850,780,900,870});
        setupLineChart(lineChart_profit, new float[]{2000,2500,2300,2800,3000});

        setupPieChart(pieChart_expenditure, new float[]{7.5f,27.5f,3.5f,42.5f}, new String[]{"Seeds","Fertilizer","Pesticides","Labor"});

        setupYieldCompareChart(barChart_yield_compare,62,61);

        // Setup TTS for each subtitle
        setupTTS(R.id.tts_crop_perf, "This is called a bar graph, This graph represents your crop yield per acre over last 5 years, the taller the graph the more the yield, to check exact values you can see number above each bar, you can check your crop yield improvement from here");
        setupTTS(R.id.tts_other_crop, "This graph is also a bar graph like before, this represents other crop average yield details over last 5 years in your region, you can check which crop yield is improving over last 5 years");
        setupTTS(R.id.tts_rainfall, "This is a line graph which shows values of average rainfall in your region, The points in the graph corresponds to the exact values of average rainfall over last 5 years in your region");
        setupTTS(R.id.tts_pest_disease, "These are the potential pest and disease risk in your region for your crop based on nearby pest reports and disease discovered. you can ask expert for immediate help");
        setupTTS(R.id.tts_profit, "This is a line graph which represents your per year profit over last 5 year, you can see if your profit is improving or not");
        setupTTS(R.id.tts_expenditure, "This is a pie graph, which is showing your expenditure division per crop cycle, you can see where your money is going and how much. you can think like 42% is same as 42 rupees if you have spent 100 rupees total");
        setupTTS(R.id.tts_yield_compare, "This bar graph shows comparison between your district average yield and your yield, you can check if your yield is more than others or not.");

        // Handle system insets
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void setupTTS(int buttonId, String englishText){
        ImageButton btn = findViewById(buttonId);
        btn.setOnClickListener(v -> speakText(englishText));
    }

    private void speakText(String englishText){
        if (englishText == null || englishText.isEmpty()) return;

        String selectedLang = translationHelper.getSelectedLanguage(); // "ta" for Tamil, "en" for English

        if ("ta".equals(selectedLang)) {
            // Translate English -> Tamil first
            translationHelper.translateText(englishText, translatedText -> {
                tts.setLanguage(new Locale("ta", "IN"));
                tts.setPitch(1.0f);
                tts.setSpeechRate(1.0f);
                tts.speak(translatedText, TextToSpeech.QUEUE_FLUSH, null, "tts_ta");
            });
        } else {
            // Speak in English
            tts.setLanguage(Locale.ENGLISH);
            tts.setPitch(1.0f);
            tts.setSpeechRate(1.0f);
            tts.speak(englishText, TextToSpeech.QUEUE_FLUSH, null, "tts_en");
        }
    }


    private void setupCropBarChart(BarChart chart, float[] values){
        ArrayList<BarEntry> entries = new ArrayList<>();
        for(int i=0;i<values.length;i++) entries.add(new BarEntry(i, values[i]));

        BarDataSet dataSet = new BarDataSet(entries,"Yield (tons/acre)");
        dataSet.setColors(
                ContextCompat.getColor(this, android.R.color.holo_green_light),
                ContextCompat.getColor(this, android.R.color.holo_red_light),
                ContextCompat.getColor(this, android.R.color.holo_blue_light),
                ContextCompat.getColor(this, android.R.color.holo_green_dark),
                ContextCompat.getColor(this, android.R.color.holo_orange_light)
        );

        BarData data = new BarData(dataSet);
        data.setBarWidth(0.9f);
        chart.setData(data);
        chart.setFitBars(true);

        String[] years = {"2019","2020","2021","2022","2023"};
        XAxis xAxis = chart.getXAxis();
        xAxis.setGranularity(1f);
        xAxis.setPosition(XAxis.XAxisPosition.BOTTOM);
        xAxis.setValueFormatter(new ValueFormatter(){
            @Override
            public String getFormattedValue(float value){
                if(value>=0 && value<years.length) return years[(int)value];
                else return "";
            }
        });

        chart.invalidate();
    }

    private void setupLineChart(LineChart chart, float[] values){
        ArrayList<Entry> entries = new ArrayList<>();
        for(int i=0;i<values.length;i++) entries.add(new Entry(i,values[i]));

        LineDataSet dataSet = new LineDataSet(entries,"Data");
        dataSet.setColor(ContextCompat.getColor(this,android.R.color.holo_green_light));
        dataSet.setCircleColor(ContextCompat.getColor(this,android.R.color.holo_red_light));

        LineData lineData = new LineData(dataSet);
        chart.setData(lineData);

        String[] years = {"2019","2020","2021","2022","2023"};
        XAxis xAxis = chart.getXAxis();
        xAxis.setGranularity(1f);
        xAxis.setPosition(XAxis.XAxisPosition.BOTTOM);
        xAxis.setValueFormatter(new ValueFormatter(){
            @Override
            public String getFormattedValue(float value){
                if(value>=0 && value<years.length) return years[(int)value];
                else return "";
            }
        });

        chart.invalidate();
    }

    private void setupPieChart(PieChart chart, float[] values, String[] labels){
        ArrayList<PieEntry> entries = new ArrayList<>();
        for(int i=0;i<values.length;i++) entries.add(new PieEntry(values[i],labels[i]));

        PieDataSet dataSet = new PieDataSet(entries,"Expenditure");
        dataSet.setColors(
                ContextCompat.getColor(this,android.R.color.holo_red_light),
                ContextCompat.getColor(this,android.R.color.holo_purple),
                ContextCompat.getColor(this,android.R.color.holo_blue_light),
                ContextCompat.getColor(this,android.R.color.holo_green_dark)
        );

        PieData data = new PieData(dataSet);
        chart.setData(data);
        chart.invalidate();
    }

    private void setupYieldCompareChart(BarChart chart, float yourYield, float districtYield){
        ArrayList<BarEntry> entries = new ArrayList<>();
        entries.add(new BarEntry(0,yourYield));
        entries.add(new BarEntry(1,districtYield));

        BarDataSet dataSet = new BarDataSet(entries,"Yield Comparison");
        dataSet.setColors(
                ContextCompat.getColor(this,android.R.color.holo_purple),
                ContextCompat.getColor(this,android.R.color.holo_green_light)
        );

        BarData data = new BarData(dataSet);
        data.setBarWidth(0.5f);
        chart.setData(data);
        chart.setFitBars(true);

        String[] labels = {"Your Yield","District Avg"};
        XAxis xAxis = chart.getXAxis();
        xAxis.setGranularity(1f);
        xAxis.setPosition(XAxis.XAxisPosition.BOTTOM);
        xAxis.setValueFormatter(new ValueFormatter(){
            @Override
            public String getFormattedValue(float value){
                if(value>=0 && value<labels.length) return labels[(int)value];
                else return "";
            }
        });

        chart.invalidate();
    }

    @Override
    protected void onDestroy(){
        if(tts!=null){
            tts.stop();
            tts.shutdown();
        }
        super.onDestroy();
    }




}
