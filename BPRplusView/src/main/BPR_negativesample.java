package main;

import java.io.IOException;
import algorithms.bpr_negativesample;
import algorithms.ItemPopularity;

public class BPR_negativesample extends main {
	public static void main(String argv[]) throws IOException {
		String method = "bpr_negativesample";
		double w0 = 2000;    // learning rate
		boolean showProgress = true;  // whether evaluate after every iter
		boolean showLoss = false;   // whether show loss after every iter						
		double alpha = 0.4;   // Popularity parameter	
		int factors = 10; 	// number of latent factors.
		int maxIter = 100; 	// maximum iterations.
		boolean adaptive = false; 	// Whether to use adaptive learning rate 
		double reg = 0.01; 	// regularization parameters
		String datafile = "data/Tmall_purchase";
		int showbound = 400;   // no evaluate outcome before showbound
		int showcount = 1;   // outcome at every showtime iter 
		int paraK = 1;   // DNS paramete
		int negnum = 1;   // negtive sample parameter
		
		if (argv.length > 0) {
			w0 = Double.parseDouble(argv[0]);
			showProgress = Boolean.parseBoolean(argv[1]);
			showLoss = Boolean.parseBoolean(argv[2]);
			factors = Integer.parseInt(argv[3]);
			maxIter = Integer.parseInt(argv[4]);
			reg = Double.parseDouble(argv[5]);
			alpha = Double.parseDouble(argv[6]);
			datafile = argv[7];
			showbound = Integer.parseInt(argv[8]);
			showcount = Integer.parseInt(argv[9]);
			paraK = Integer.parseInt(argv[10]);
			negnum = Integer.parseInt(argv[11]);
		}

		ReadRatings_HoldOneOut(datafile);		
		System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%.6f, w0=%.6f, alpha=%.4f,paraK=%d,negnum = %d\\n",
				method, showProgress, factors, maxIter, reg, w0, alpha, paraK,negnum);
		System.out.println("====================================================");		
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		evaluate_model(popularity, "Popularity");		
		double init_mean = 0;
		double init_stdev = 0.01;
				
		bpr_negativesample bpr = new bpr_negativesample(trainMatrix, testRatings, topK, threadNum, 
				factors, maxIter, w0, false, reg, init_mean, init_stdev, showProgress,showbound,showcount,paraK,negnum);
		evaluate_model(bpr, "bpr_negativesample");
			
	} // end main
}
