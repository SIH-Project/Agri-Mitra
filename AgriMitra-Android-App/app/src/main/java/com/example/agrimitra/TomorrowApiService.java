package com.example.agrimitra;

import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Query;

public interface TomorrowApiService {
    @GET("v4/timelines")
    Call<TomorrowResponse> getDailyForecast(
            @Query("location") String location,
            @Query("fields") String fields,
            @Query("timesteps") String timesteps,
            @Query("units") String units,
            @Query("apikey") String apiKey
    );
}
