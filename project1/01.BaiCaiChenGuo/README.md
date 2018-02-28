Just launch the "Project 1.ipynb" ipython notebook, then you can view our report (plus code) in your browser via localhost.

Note for data hierarchy:
You need to put the Raphael dataset in the following way to make our code work:
- ./data2/
   - raphael/
      - xxx.jpg/tif
      ...
      - xxx.jpg/tif
   - notRaphael/
      - xxx.jpg/tif
      ...
      - xxx.jpg/tif

- ./test/
   - xxx.jpg/tif
   ...
   - xxx.jpg/tif

In other words, put the real drawings under ./data2/raphael/, the fake drawings under ./data2/notRaphael, and the unknown drawings under ./test/
