package com.dugking.measure;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
 
/**
 * DATE: 16-6-18 TIME: ÉÏÎç10:00
 */
 
/**
 * ²Î¿¼ÎÄÏ×£ºhttp://www-nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
 */
public class NormalizedMutualInformation {
    static String path = "src/data.txt";
 
    static void loadData(List<List<Integer>> lists) {
        try {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                String[] data = line.split("\\s+");
                ArrayList<Integer> integers = new ArrayList<>();
                for (String s : data) {
                    integers.add(Integer.parseInt(s));
                }
                lists.add(integers);
            }
            bufferedReader.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
 
    public static void main(String[] args) {
        List<List<Integer>> lists = new ArrayList<>();
        loadData(lists);
        int K = lists.size();
        int N = 0;
        int[] clusters = new int[K];
        for (int i = 0; i < K; i++) {
            clusters[i] = lists.get(i).size();
            N += clusters[i];
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (List<Integer> list : lists) {
            for (Integer integer : list) {
                map.put(integer, map.getOrDefault(integer, 0) + 1);
            }
        }
        double clusterEntropy = 0;
        for (int cluster : clusters) {
            double tmp = 1.0 * cluster / N;
            clusterEntropy -= (tmp * (Math.log(tmp) / Math.log(2)));
        }
        System.out.println("clusterEntropy = " + clusterEntropy);
        double classEntropy = 0;
        for (Integer integer : map.values()) {
            double tmp = 1.0 * integer / N;
            classEntropy -= (tmp * (Math.log(tmp) / Math.log(2)));
        }
        System.out.println("classEntropy = " + classEntropy);
        double totalEntropy = 0;
        Map<Integer, Integer> tmpMap = new HashMap<>();
        for (int i = 0; i < K; i++) {
            int wk = clusters[i];
            tmpMap.clear();
            for (Integer integer : lists.get(i)) {
                tmpMap.put(integer, tmpMap.getOrDefault(integer, 0) + 1);
            }
            for (Map.Entry<Integer, Integer> entry : tmpMap.entrySet()) {
                int cj = map.get(entry.getKey());
                int value = entry.getValue();
                totalEntropy += (1.0 * value / N * (Math.log(1.0 * N * value / (wk * cj)) / Math.log(2)));
            }
        }
        System.out.println("totalEntropy = " + totalEntropy);
        double nmi = 2 * totalEntropy / (clusterEntropy + classEntropy);
        System.out.println("nmi = " + nmi);
    }
}
