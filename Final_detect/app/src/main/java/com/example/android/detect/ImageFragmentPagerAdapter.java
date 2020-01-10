package com.example.android.detect;

import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentPagerAdapter;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import static com.example.android.detect.MainActivity.Image_name;
import static com.example.android.detect.MainActivity.number_item;

class ImageFragmentPagerAdapter extends FragmentPagerAdapter {

    public ImageFragmentPagerAdapter(FragmentManager fm){
        super(fm);
    }
    @Override
    public int getCount(){
        return number_item;
    }
    @Override
    public Fragment getItem(int position){
        SwipeFragment fragment=new SwipeFragment();

        return SwipeFragment.newInstance(position);
    }


}
