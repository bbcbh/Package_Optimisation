package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.compress.archivers.sevenz.SevenZArchiveEntry;
import org.apache.commons.compress.archivers.sevenz.SevenZFile;
import org.apache.commons.compress.archivers.sevenz.SevenZOutputFile;
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
			String[] new_param_header, HashMap<String, double[]> default_param_range,
			HashMap<String, String> cross_ref_map) throws FileNotFoundException, IOException {

		HashMap<String, ArrayList<Number>> map_val = new HashMap<>();	
		int numExtracted  = 0;

		if (baseSeedList != null && baseSeedList.exists()) {
			String[] baseSeedLine = extracted_lines_from_text(baseSeedList);
			String[] baseSeedHeader = baseSeedLine[0].split(",");

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
			numExtracted = baseSeedLine.length - 1;
			System.out.printf("# seed from %s = %d\n", baseSeedList.getAbsolutePath(), numExtracted);
		}

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
					if(numExtracted != 0) {
					System.out.printf(
							" Parameter %s not found from original list, attempt to sample from default_param_range instead.\n",
							colName);
					}
					if (!default_param_range.containsKey(colName)) {
						System.out.printf("Error! Parameter %s not found in default_param_range. Exiting.\n",
								colName);
						System.exit(-1);
					} else {
						raw_val.put(colName,
								extractedWeightedInterpolateFunction(pol, default_param_range.get(colName)));
					}
				} else {
					double[] val = new double[ent.size()];
					for (int i = 0; i < val.length; i++) {
						val[i] = ent.get(i).doubleValue();

						double ref_value = 1;
						if (cross_ref_map.containsKey(colName)) {
							ref_value = map_val.get(cross_ref_map.get(colName)).get(i).doubleValue();
							val[i] = val[i] / ref_value;
						}

						if (default_param_range.containsKey(colName)) {
							double[] range = default_param_range.get(colName);
							if (!(range[0] <= val[i] && val[i] <= range[1])) {
								double org_val = val[i];
								val[i] = range[0] + RNG.nextDouble() * (range[1] - range[0]);
								System.out.printf("Parameter %s at Row #%d resampled %f->%f to be within %s.\n",
										colName, i, org_val, val[i], Arrays.toString(range));

								ent.set(i, val[i] * ref_value);

							}
						}

					}
					raw_val.put(colName, extractedWeightedInterpolateFunction(pol, val));
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
				Number[] val = new Number[new_param_header.length];
				boolean[] directCopy = new boolean[new_param_header.length];
				Arrays.fill(directCopy, false);

				if (copied_entry < numExtracted) {
					// Copy entry
					for (int p = 0; p < val.length; p++) {
						String colName = new_param_header[p];
						if (map_val.containsKey(colName)) {
							val[p] = map_val.get(colName).get(copied_entry);
							directCopy[p] = true;
						} else {
							val[p] = raw_val.get(colName).value(RNG.nextDouble());
						}
					}
					copied_entry++;
				} else {
					// Resample entry
					for (int p = 0; p < val.length; p++) {
						String colName = new_param_header[p];
						if (raw_val.containsKey(colName)) {
							val[p] = raw_val.get(colName).value(RNG.nextDouble());
						} else {
							directCopy[p] = true;
							ArrayList<Number> ent = map_val.get(colName);
							val[p] = ent.get(RNG.nextInt(ent.size()));

						}
					}
				}

				HashMap<String, Number> cross_ref_val = new HashMap<>();
				for (int p = 0; p < val.length; p++) {
					String colName = new_param_header[p];
					cross_ref_val.put(colName, val[p]);
				}

				// Check for cross ref
				for (int p = 0; p < val.length; p++) {
					if (!directCopy[p]) {
						String colName = new_param_header[p];
						if (cross_ref_map.containsKey(colName)) {
							val[p] = val[p].doubleValue() * cross_ref_val.get(cross_ref_map.get(colName)).doubleValue();
						}
					}
				}

				for (int p = 0; p < val.length; p++) {
					if (paramline.length() != 0) {
						paramline.append(',');
					}
					paramline.append(val[p]);
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
	
	public static void zipFile(File[] FileList, File tarFile, boolean rmSource)
			throws IOException, FileNotFoundException {
		SevenZOutputFile outputZip = new SevenZOutputFile(tarFile);

		SevenZArchiveEntry entry;
		FileInputStream fIn;

		for (int fI = 0; fI < FileList.length; fI++) {
			entry = outputZip.createArchiveEntry(FileList[fI], FileList[fI].getName());
			outputZip.putArchiveEntry(entry);
			fIn = new FileInputStream(FileList[fI]);
			outputZip.write(fIn);
			outputZip.closeArchiveEntry();
			fIn.close();
		}

		outputZip.close();

		// Clean up
		if (rmSource) {
			for (File f : FileList) {
				f.delete();
			}
		}
	}
	
	public static HashMap<String, ArrayList<String[]>> extractedLinesFrom7Zip(File zipFile,
			HashMap<String, ArrayList<String[]>> zip_ent, Pattern keyPattern) throws IOException {
		SevenZFile inputZip = new SevenZFile(zipFile);
		SevenZArchiveEntry inputEnt;
		final int BUFFER = 2048;

		byte[] buf = new byte[BUFFER];
		while ((inputEnt = inputZip.getNextEntry()) != null) {
			String file_name = inputEnt.getName();
			StringBuilder str_builder = new StringBuilder();
			String line;
			ArrayList<String[]> lines = new ArrayList<>();
			int count;
			while ((count = inputZip.read(buf, 0, BUFFER)) != -1) {
				str_builder.append(new String(Arrays.copyOf(buf, count)));
			}
			BufferedReader reader = new BufferedReader(new StringReader(str_builder.toString()));
			while ((line = reader.readLine()) != null) {
				if (line.length() > 0) {
					lines.add(line.split(","));
				}
			}

			String key = file_name;
			if (keyPattern != null) {
				Matcher m = keyPattern.matcher(file_name);
				if (m.find()) {
					if (m.groupCount() > 0) {
						key = m.group(1);
					} else {
						System.err.print(
								"extractedLinesFrom7Zip: Matcher has no group - using filename as key instead.\n");
					}
				} else {
					// System.err.printf("extractedLinesFrom7Zip: File entries %s does not match
					// with pattern "
					// + "- using filename as key instead.\n", file_name);
				}
			}

			zip_ent.put(key, lines);
		}
		inputZip.close();
		return zip_ent;
	}

}
