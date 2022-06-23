//Accepts a range of ROOT histogram plots of the form
//2dplots_{energy}_keVgamma.root and converts their bin contents to a single-line CSV file.

{
  for (int i=1; i<10000; i+=1)
    {
      //Define output name
      ofstream fileout(Form("path/to/output/2dplot_%d_keVgamma.csv", i));
      
      //Open ROOT Histogram
      TFile *f = new TFile(Form("path/to/input/2dplots_%d_keVgamma.root", i));

      //cout << "got file " << i << endl;
      
      //Get TH2D from the ROOT Histogram
      TH2D *hsum = (TH2D*)f->Get("sum");
      
      //Get Number of bins from the histograms
      int nbinsx=hsum->GetXaxis()->GetNbins();
      int nbinsy=hsum->GetXaxis()->GetNbins();

      for (int x=1; x<=nbinsx; x++)
        {
	  for (int y=1; y<=nbinsy; y++)
            {
	      fileout<<hsum->GetBinContent(x,y)<<",";
	      
            }
        }

   

      fileout<<endl;

      delete hsum;
      delete f;
      
    }
}
