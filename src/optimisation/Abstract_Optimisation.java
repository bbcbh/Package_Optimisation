package optimisation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateFunctionMappingAdapter;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.NelderMeadSimplex;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.PowellOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer;

import random.MersenneTwisterRandomGenerator;
import util.StaticMethods;

public abstract class Abstract_Optimisation {

//	  Optimizer Type 			Best Use Case 		Performance (Speed) 
//	  CMAES 	Stochastic 		Global/Non-smooth 	Slower 
//	  BOBYQA 	Quadratic Appx 	Smooth/Bounded 		Moderate/Fast
//	  Simplex 	Direct Search 	Local/Non-smooth 	Moderate 
//	  Powell 	Direct Search 	Local/Smooth 		Fast 

	public static final int OPT_TYPE_SIMPLEX = 0;
	public static final int OPT_TYPE_CMAES = 1;
	public static final int OPT_TYPE_BOBYQA = 2;
	public static final int OPT_TYPE_POWELL = 3;

	// Parameter setting
	protected final String[] param_to_opt;
	protected final double[][] param_boundaries;
	protected final HashMap<String, String> cross_ref_map;

	// Optimiser setting
	protected int optType = OPT_TYPE_SIMPLEX;
	protected final int[] opt_time_range;
	protected final double[][] opt_setting;
	protected final String[] opt_outcome_csv;
	protected double opt_rel_tol = 1e-5;
	protected double opt_abs_tol = 1e-10;
	protected MaxEval opt_maxVal = MaxEval.unlimited();

	// File paths
	protected final String path_dirName;
	protected final String path_seed;
	protected final File file_seed_file;
	protected final String[] seed_file_lines;
	protected final String[] seed_file_header;
	protected static final String OPTDIR_FORMAT = "%s_%d";

	// Default setting
	protected long opt_rng_seed = 2251912207291119l;
	protected int opt_feasible_count = 10;
	protected double opt_sigma_common = 0.1;

	public static final String fileformat_output_txt = "Output.txt";
	public static final String fileformat_point_cache = "OptProgress_PointCache_%s.csv";
	public static final String fileformat_opt_outcomes = "OptProgress_ParamList_%s.csv";

	public Abstract_Optimisation(String dirName, String seed_name) throws IOException {
		this.path_dirName = dirName;
		this.path_seed = seed_name;

		File file_opt_setting = new File(dirName, "optSetting.prop");
		FileInputStream fIS = new FileInputStream(file_opt_setting);
		Properties prop = new Properties();
		prop.loadFromXML(fIS);
		fIS.close();
		String param_to_opt_str = prop.getProperty("PROP_PARAM_TO_OPT");

		param_to_opt = param_to_opt_str.split(",");
		HashMap<String, double[]> default_sample_range = new HashMap<>();
		cross_ref_map = new HashMap<>();

		for (String param : param_to_opt) {
			String ent = prop.getProperty(String.format("PROP_PARAM_SETTING_%s", param));
			if (ent != null) {
				ent.replaceAll("\\s", "");
				String[] sp = ent.split(",");
				default_sample_range.put(param, new double[] { Double.parseDouble(sp[0]), Double.parseDouble(sp[1]) });
				if (sp.length > 2) {
					cross_ref_map.put(param, sp[2]);
				}
			}
		}

		opt_time_range = (int[]) util.PropValUtils.propStrToObject(prop.getProperty("PROP_OPT_TIME_RANGE"),
				int[].class);

		opt_setting = (double[][]) util.PropValUtils.propStrToObject(prop.getProperty("PROP_OPT_SETTING"),
				double[][].class);

		if (prop.containsKey("PPOP_OPT_TOLERANCE")) {
			double[] opt_tol = (double[]) util.PropValUtils.propStrToObject(prop.getProperty("PPOP_OPT_TOLERANCE"),
					double[].class);
			opt_rel_tol = opt_tol[0];
			opt_abs_tol = opt_tol[1];
		}

		if (prop.containsKey("PROP_OPT_MAX_EVAL")) {
			opt_maxVal = new MaxEval(Integer.parseInt(prop.getProperty("PROP_OPT_MAX_EVAL")));
		}

		if (prop.containsKey("PROP_OPT_RNG_SEED")) {
			opt_rng_seed = Long.parseLong(prop.getProperty("PROP_OPR_RNG_SEED"));
		}

		if (prop.containsKey("PROP_OPT_SIGMA")) {
			opt_sigma_common = Double.parseDouble(prop.getProperty("PROP_OPT_SIGMA"));
		}

		// Set up initial parameter array
		File seed_file_test = new File(dirName, path_seed);
		if (seed_file_test.isDirectory()) {
			file_seed_file = new File(new File(dirName, path_seed), String.format("%s.csv", path_seed));
		} else {
			file_seed_file = seed_file_test;
		}
		seed_file_lines = util.StaticMethods.extracted_lines_from_text(file_seed_file);
		seed_file_header = seed_file_lines[0].split(",");

		param_boundaries = new double[2][param_to_opt.length];
		for (int i = 0; i < param_to_opt.length; i++) {
			double[] range = default_sample_range.get(param_to_opt[i]);
			param_boundaries[0][i] = range[0];
			param_boundaries[1][i] = range[1];
		}

		opt_outcome_csv = prop.getProperty("PROP_OPT_OUTCOME_CSV").replaceAll("\\s", "").split(",");

	}

	protected abstract MultivariateFunction generateObjectiveFunc(int seed_row, String[] seed_file_def_val);

	public void setOptType(int optType) {
		this.optType = optType;
	}

	public void runOptimisation() {

		boolean hasReplacement = false;
		for (int p = 1; p < seed_file_lines.length; p++) {
			File preResult = new File(new File(path_dirName), String.format(
					Abstract_Optimisation.fileformat_opt_outcomes, String.format(OPTDIR_FORMAT, path_seed, p - 1)));
			if (preResult.exists()) {
				try {
					String[] pre_lines = util.StaticMethods.extracted_lines_from_text(preResult);
					double minR = Double.POSITIVE_INFINITY;
					for (int i = 1; i < pre_lines.length; i++) {
						String[] pre_line_ent = pre_lines[i].split(",");
						double residue = Double.parseDouble(pre_line_ent[pre_line_ent.length - 1]);
						if (residue < minR) {
							hasReplacement = true;
							StringBuilder sb = new StringBuilder();
							for (int c = 0; c < seed_file_header.length; c++) {
								if (c > 0) {
									sb.append(',');
								}
								sb.append(pre_line_ent[c]);
							}
							seed_file_lines[p] = sb.toString();
							minR = residue;
						}
					}
					System.out.printf("%s: Initial value replaced with:\n   [%s] with residue of %f.\n",
							preResult.getName(), seed_file_lines[p], minR);

				} catch (IOException ex) {
					System.err.printf("Warning! %s encountered in reading %s. Using default parameter instead.\n",
							ex.toString(), preResult.getAbsolutePath());
					ex.printStackTrace(System.err);
				}
			}
		}
		if (hasReplacement) {
			try {
				Files.copy(file_seed_file.toPath(),
						new File(file_seed_file.getParent(), String.format("org_%s", file_seed_file.getName()))
								.toPath(), StandardCopyOption.REPLACE_EXISTING);

				PrintWriter pWri_seed = new PrintWriter(file_seed_file);
				for (int i = 0; i < seed_file_lines.length; i++) {
					pWri_seed.println(seed_file_lines[i]);
				}
				pWri_seed.close();

			} catch (Exception e) {
				e.printStackTrace(System.err);
			}

		}

		ExecutorService exec = Executors.newFixedThreadPool(seed_file_lines.length - 1);
		for (int seed_row = 1; seed_row < seed_file_lines.length; seed_row++) {
			String[] seed_file_def_val = seed_file_lines[seed_row].split(",");
			HashMap<String, Double> init_value = new HashMap<>();
			for (int i = 0; i < seed_file_header.length; i++) {
				init_value.put(seed_file_header[i], Double.valueOf(seed_file_def_val[i]));
			}
			double[] param_init = new double[param_to_opt.length];
			for (int i = 0; i < param_to_opt.length; i++) {
				param_init[i] = init_value.get(param_to_opt[i]).doubleValue();

			}
			// Adjust for cross reference
			for (int i = 0; i < param_to_opt.length; i++) {
				if (cross_ref_map.containsKey(param_to_opt[i])) {
					param_init[i] = param_init[i] / init_value.get(cross_ref_map.get(param_to_opt[i]));
				}
			}

			// Objective function
			MultivariateFunction func = generateObjectiveFunc(seed_row, seed_file_def_val);

			// Set up optimiser

			String wk_dir_name = String.format(OPTDIR_FORMAT, path_seed, seed_row - 1);
			MultivariateFunctionMappingAdapter wrapper = new MultivariateFunctionMappingAdapter(func,
					param_boundaries[0], param_boundaries[1]);

			ObjectiveFunction objFunc = new ObjectiveFunction(wrapper);

			Runnable runnable_opt = new Runnable() {
				@Override
				public void run() {
					InitialGuess initial_guess;
					MultivariateOptimizer optimizer;

					initial_guess = new InitialGuess(wrapper.boundedToUnbounded(param_init));

					PointValuePair pV;
					System.out.printf("%s : Optimisation start Max Eval = %d.\n", wk_dir_name, opt_maxVal.getMaxEval());

					switch (optType) {

					case OPT_TYPE_SIMPLEX:
						final NelderMeadSimplex simplex;

						simplex = new NelderMeadSimplex(param_init.length);
						optimizer = new SimplexOptimizer(opt_rel_tol, opt_abs_tol);

						try {
							pV = optimizer.optimize(objFunc, simplex, GoalType.MINIMIZE, initial_guess, opt_maxVal);
							double[] point = wrapper.unboundedToBounded(pV.getPoint());

							StringBuilder pt_str = new StringBuilder();
							for (double pt : point) {
								if (pt_str.length() != 0) {
									pt_str.append(',');
								}
								pt_str.append(String.format("%.5f", pt));
							}

							System.out.printf("%s :Simplex Optimisation Completed.\nP = [%s], V = %f\n", wk_dir_name,
									pt_str.toString(), pV.getValue());

						} catch (org.apache.commons.math3.exception.TooManyEvaluationsException ex) {
							System.out.printf(
									"%s :Simplex Optimisation Eval limit of (ex.getMax=%d) reached.\nSimplex (bounded):\n",
									wk_dir_name, ex.getMax());

							PointValuePair[] res = simplex.getPoints();
							Arrays.sort(res, new Comparator<PointValuePair>() {
								@Override
								public int compare(PointValuePair o1, PointValuePair o2) {
									return Double.compare(o1.getValue(), o2.getValue());
								}

							});

							for (PointValuePair pV_I : res) {
								double[] point = wrapper.unboundedToBounded(pV_I.getPoint());

								StringBuilder pt_str = new StringBuilder();
								for (double pt : point) {
									if (pt_str.length() != 0) {
										pt_str.append(',');
									}
									pt_str.append(String.format("%.5f", pt));
								}

								System.out.printf("%s :P = [%s], V = %f\n", wk_dir_name, pt_str.toString(),
										pV_I.getValue());

							}

						}
						break;
					case OPT_TYPE_CMAES:
						optimizer = new CMAESOptimizer(opt_maxVal.getMaxEval(), // maxIterations
								0.0, // stopFitness (threshold to stop)
								true, // isActiveCMA
								0, // diagonalOnly (iterations with diagonal covariance)
								opt_feasible_count, // checkFeasableCount
								new MersenneTwisterRandomGenerator(opt_rng_seed), // random generator
								false, // generateStatistics
								new SimpleValueChecker(opt_rel_tol, opt_abs_tol) // convergence checker
						);
						try {
							double[] sig_val = new double[param_to_opt.length];
							Arrays.fill(sig_val, opt_sigma_common);

							SimpleBounds bounds = new SimpleBounds(wrapper.boundedToUnbounded(param_boundaries[0]),
									wrapper.boundedToUnbounded(param_boundaries[1]));

							pV = optimizer.optimize(opt_maxVal, objFunc, GoalType.MINIMIZE, initial_guess,
									new CMAESOptimizer.Sigma(sig_val), bounds, // Sigma and bound
									new CMAESOptimizer.PopulationSize(opt_feasible_count));
							double[] point = wrapper.unboundedToBounded(pV.getPoint());

							StringBuilder pt_str = new StringBuilder();
							for (double pt : point) {
								if (pt_str.length() != 0) {
									pt_str.append(',');
								}
								pt_str.append(String.format("%.5f", pt));
							}

							System.out.printf("%s :CMAES Optimisation Completed.\nP = [%s], V = %f\n", wk_dir_name,
									pt_str.toString(), pV.getValue());
						} catch (org.apache.commons.math3.exception.TooManyEvaluationsException ex) {
							System.out.printf("%s :CMAES Optimisation Eval limit of (ex.getMax=%d) reached\n",
									wk_dir_name, ex.getMax());
						}

						break;
					case OPT_TYPE_BOBYQA:
						// For a problem of dimension n, its value must be in the interval [n+2,
						// (n+1)(n+2)/2].
						// Choices that exceed 2n+1 are not recommended.
						int interpolationPoints = 2 * param_to_opt.length + 1;
						optimizer = new BOBYQAOptimizer(interpolationPoints);

						SimpleBounds bounds = new SimpleBounds(wrapper.boundedToUnbounded(param_boundaries[0]),
								wrapper.boundedToUnbounded(param_boundaries[1]));

						try {
							pV = optimizer.optimize(opt_maxVal, // Termination criteria: max evaluations
									objFunc, // The function to minimize
									GoalType.MINIMIZE, // Optimization goal
									initial_guess, // Starting point
									bounds // Required for BOBYQA
							);

							double[] point = wrapper.unboundedToBounded(pV.getPoint());

							StringBuilder pt_str = new StringBuilder();
							for (double pt : point) {
								if (pt_str.length() != 0) {
									pt_str.append(',');
								}
								pt_str.append(String.format("%.5f", pt));
							}

							System.out.printf("%s :BOBYQA Optimisation Optimisation Completed.\nP = [%s], V = %f\n",
									wk_dir_name, pt_str.toString(), pV.getValue());

						} catch (org.apache.commons.math3.exception.TooManyEvaluationsException ex) {
							System.out.printf("%s :BOBYQA Optimisation Eval limit of (ex.getMax=%d) reached\n",
									wk_dir_name, ex.getMax());
						}

						break;
					case OPT_TYPE_POWELL:
						optimizer = new PowellOptimizer(opt_rel_tol, opt_abs_tol);

						try {
							pV = optimizer.optimize(opt_maxVal, // Maximum evaluations
									objFunc, // The function to optimize
									GoalType.MINIMIZE, // Optimization goal
									initial_guess // Starting point
							);

							double[] point = wrapper.unboundedToBounded(pV.getPoint());

							StringBuilder pt_str = new StringBuilder();
							for (double pt : point) {
								if (pt_str.length() != 0) {
									pt_str.append(',');
								}
								pt_str.append(String.format("%.5f", pt));
							}

							System.out.printf("%s :POWELL Optimisation Optimisation Completed.\nP = [%s], V = %f\n",
									wk_dir_name, pt_str.toString(), pV.getValue());

						} catch (org.apache.commons.math3.exception.TooManyEvaluationsException ex) {
							System.out.printf("%s :POWELL Optimisation Eval limit of (ex.getMax=%d) reached\n",
									wk_dir_name, ex.getMax());
						}

						break;
					default:
						System.err.printf("Error! Opt_Type = %d not defined/implemented. Exiting.\n", optType);
						System.exit(-1);

					} // End switch(optType) {...}

				} // End run()
			};

			if (seed_file_lines.length == 2) {
				runnable_opt.run();
			} else {
				exec.submit(runnable_opt);
			}

		}

		if (seed_file_lines.length > 2) {
			exec.shutdown();
			try {
				if (!exec.awaitTermination(2, TimeUnit.DAYS)) {
					System.err.println("Thread time-out!");
				}
			} catch (InterruptedException e) {
				e.printStackTrace(System.err);
			}
		}

	}

	protected static void generateResidueOutcomeFiles(File file_base_dir, String sim_id, String[] seed_header,
			String[] seed_val_str, double[] point, double residue) throws FileNotFoundException, IOException {
		File file_outcome = new File(file_base_dir, String.format(fileformat_opt_outcomes, sim_id));

		PrintWriter pWri_outcome;
		if (!file_outcome.exists()) {
			pWri_outcome = new PrintWriter(file_outcome);
			StaticMethods.writeEntries(pWri_outcome, seed_header);
			pWri_outcome.println(",,RES");
		} else {
			pWri_outcome = new PrintWriter(new FileWriter(file_outcome, true));
		}
		StaticMethods.writeEntries(pWri_outcome, seed_val_str);
		pWri_outcome.print(",,");
		pWri_outcome.print(residue);
		pWri_outcome.println();
		pWri_outcome.close();

		File file_pointCache = new File(file_base_dir, String.format(fileformat_point_cache, sim_id));
		PrintWriter pWri_pointCache = new PrintWriter(new FileWriter(file_pointCache, true));
		pWri_pointCache.printf("%s:%f\n", Arrays.toString(point), residue);
		pWri_pointCache.close();
	}

}
