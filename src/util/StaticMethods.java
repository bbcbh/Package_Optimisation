package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.interpolation.UnivariateInterpolator;

import random.RandomGenerator;

public class StaticMethods {
	public static UnivariateFunction extractedWeightedInterpolateFunction(UnivariateInterpolator polator,
			double[] dist_val) {
		Arrays.sort(dist_val);
		if (Double.isNaN(dist_val[dist_val.length - 1])) {
			return null;
		}

		double[] x_val = new double[dist_val.length];
		for (int x = 1; x < dist_val.length; x++) {
			x_val[x] = x_val[x - 1] + 1.0 / dist_val.length;
		}
		x_val[x_val.length - 1] = 1;

		UnivariateFunction resFunc = polator.interpolate(x_val, dist_val);
		return resFunc;
	}
	
	
	public static String[] extracted_lines_from_text(File srcTxt) throws FileNotFoundException, IOException {
		ArrayList<String> lines = new ArrayList<>();
		BufferedReader reader = new BufferedReader(new FileReader(srcTxt));
		String line;
		int line_counter = 0;
		while ((line = reader.readLine()) != null) {
			lines.add(line);
			line_counter++;
		}
		reader.close();
		String[] line_pbs_arr = lines.toArray(new String[line_counter]);
		return line_pbs_arr;
	}


	public static void genOptSeedList(File propDir, File baseSeedList, int maxList, int numPerList, RandomGenerator RNG,
			String[] new_param_header, Map<String, double[]> map_adjustment) throws FileNotFoundException, IOException {
	
		String[] baseSeedLine = extracted_lines_from_text(baseSeedList);
		String[] baseSeedHeader = baseSeedLine[0].split(",");
		HashMap<String, ArrayList<Number>> map_val = new HashMap<>();
	
		for (int r = 1; r < baseSeedLine.length; r++) {
			String[] rowEnt = baseSeedLine[r].split(",");
			for (int c = 0; c < rowEnt.length; c++) {
				String colName = baseSeedHeader[c];
				if (!colName.isEmpty()) {
					ArrayList<Number> ent = map_val.get(colName);
					if (ent == null) {
						ent = new ArrayList<>();
						map_val.put(colName, ent);
					}
					if (colName.endsWith("SEED")) {
						ent.add(Long.valueOf(rowEnt[c]));
					} else {
						try {
							ent.add(Double.valueOf(rowEnt[c]));
						} catch (NumberFormatException ex) {							
							ent.add(Double.NaN);
						}
					}
				}
			}
		}
	
		System.out.printf("# seed from %s = %d\n", baseSeedList.getAbsolutePath(), baseSeedLine.length - 1);
	
		HashMap<String, UnivariateFunction> raw_val = new HashMap<>();
		UnivariateInterpolator pol = new LinearInterpolator();
	
		StringBuilder seedHeader = new StringBuilder();
		for (String colName : new_param_header) {
			if (seedHeader.length() != 0) {
				seedHeader.append(',');
			}
			seedHeader.append(colName);
	
			if (!colName.endsWith("SEED")) {
				ArrayList<Number> ent = map_val.get(colName);
				if (ent == null) {
					System.out.printf(
							"Warning! Parameter %s not found from orginal list, attempt to sample from map_adjustment instead.\n",
							colName);
					if (!map_adjustment.containsKey(colName)) {
						System.out.printf("Error! Parameter %s not found in map_adjustment either. Exiting.\n",
								colName);
						System.exit(-1);
					} else {
						raw_val.put(colName, extractedWeightedInterpolateFunction(pol,
								map_adjustment.get(colName)));
					}
				} else {
					double[] val = new double[ent.size()];
					double[] val_range = new double[] { Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY };
					for (int i = 0; i < val.length; i++) {
						val[i] = ent.get(i).doubleValue();
						val_range[0] = Math.min(val_range[0], val[i]);
						val_range[1] = Math.max(val_range[1], val[i]);
					}
					if (map_adjustment.containsKey(colName)) {
						double[] adj_range = map_adjustment.get(colName);
						if (val_range[1] > adj_range[1] || val_range[0] < adj_range[0]) {
							System.out.printf(
									"Parameter %s exceed pre-define range and will be adjusted to fitted to %s.\n",
									colName, Arrays.toString(adj_range));
							for (int i = 0; i < val.length; i++) {
								val[i] = adj_range[0] + (adj_range[1] - adj_range[0]) * (val[i] - val_range[0])
										/ (val_range[1] - val_range[0]);
								
								ent.set(i, val[i]);																
							}
						}
					}
					raw_val.put(colName,
							extractedWeightedInterpolateFunction(pol, val));
				}
	
			}
		}
		// Copy/Generate seed list
		PrintWriter pWri_seed_all = new PrintWriter(new File(propDir, "optSeedListAll.csv"));
		pWri_seed_all.printf("%s,,LOC\n", seedHeader.toString());
		
		PrintWriter pWri_seed_name = new PrintWriter(new File(propDir, "optSeedListName.txt"));
		pWri_seed_name.printf("# SeedList = %d\n", maxList);
	
		int seedListNum = 0;
		int copied_entry = 0;
		while (seedListNum < maxList) {
			File seedDir = new File(propDir, String.format("Seed_List_%03d", seedListNum));
			seedDir.mkdirs();
			File seedList = new File(seedDir, String.format("Seed_List_%03d.csv", seedListNum));
			
			pWri_seed_name.printf("   %s\n", seedDir.getName());
			
			
			PrintWriter pWri_seed = new PrintWriter(seedList);
			pWri_seed.println(seedHeader.toString());
			for (int i = 0; i < numPerList; i++) {
				StringBuilder paramline = new StringBuilder();
	
				if (copied_entry < baseSeedLine.length - 1) {
					// Copy entry
					for (String colName : new_param_header) {
						if (paramline.length() != 0) {
							paramline.append(',');
						}
						if (map_val.containsKey(colName)) {
							paramline.append(map_val.get(colName).get(copied_entry).toString());
						} else {
							paramline.append(raw_val.get(colName).value(RNG.nextDouble()));
						}
					}
					copied_entry++;
				} else {
					// Resample entry
					for (String colName : new_param_header) {
						if (paramline.length() != 0) {
							paramline.append(',');
						}
						if (raw_val.containsKey(colName)) {
							paramline.append(raw_val.get(colName).value(RNG.nextDouble()));
						} else {
							ArrayList<Number> ent = map_val.get(colName);
							paramline.append(ent.get(RNG.nextInt(ent.size())));
						}
					}
	
				}
				pWri_seed.println(paramline.toString());
				pWri_seed_all.printf("%s,,%s:%d\n", paramline.toString(), seedDir.getName(), i);
	
			}
	
			pWri_seed.close();
	
			seedListNum++;
		}
	
		pWri_seed_all.close();
		pWri_seed_name.close();
	
		System.out.printf("Seed List copied/generated at %s.\n", propDir);
	
	}


	public static void writeEntries(PrintWriter pWri_seed, String[] arr) {
		for (int i = 0; i < arr.length; i++) {
			if (i != 0) {
				pWri_seed.append(',');
			}
			pWri_seed.append(arr[i]);
		}
	
	}

}
