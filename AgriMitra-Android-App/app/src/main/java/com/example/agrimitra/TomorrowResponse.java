package com.example.agrimitra;

import com.google.gson.annotations.SerializedName;

import java.util.List;

public class TomorrowResponse {
    @SerializedName("data")
    public Data data;

    public static class Data {
        @SerializedName("timelines")
        public List<Timeline> timelines;
    }

    public static class Timeline {
        @SerializedName("timestep")
        public String timestep;

        @SerializedName("intervals")
        public List<Interval> intervals;
    }

    public static class Interval {
        @SerializedName("startTime")
        public String startTime;

        @SerializedName("values")
        public Values values;
    }

    public static class Values {
        @SerializedName("precipitationIntensity")
        public double precipitationIntensity;

        @SerializedName("precipitationProbability")
        public double precipitationProbability;
    }
}
