This is a first submission of 'tfaddons' 0.9.1

## Test environments

* local install: Mac OS Catalina, R 3.6.2
* Github actions: windows-latest
* Github actions: macOS-latest
* Github actions: ubuntu-16.04


## R CMD check results

There were no ERRORs, WARNINGs. There was 1 NOTE:

```
checking Rd line widths ... NOTE
  Rd file 'callback_tqdm_progress_bar.Rd':
    \usage lines wider than 90 characters:
           overall_bar_format = "{l_bar}{bar} {n_fmt}/{total_fmt} ETA: {remaining}s,  {rate_fmt}{postfix}",
  
  These lines will be truncated in the PDF manual.
```
