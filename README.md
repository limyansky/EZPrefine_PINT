# EZPrefine_PINT
An attempt to implement EZPrefine (by Mario) using the framework of PINT.

EZPrefine was made for refining pulsar candidates in fermi data, and operates
based off of MCMC.

PINT is the latest implimentation pulsar timing, this time in python. 

I use both tools individually, but they seem incompatable. The significances I
get from EZPrefine never quite match what I get in PINT. 

PINT has a MCMC fitter, but it doesn't appear to natively support photon
weights, at least in the provided sample code. Further, the sample code
does some sort of phase optimization, which requires a gaussian template that
looks like your pulse profile. Thus, including weights means also modifying the 
gaussian template code (not included in PINT, I think) to accept photon weights.
Not to mention, my timing solution isn't always good enough to see a pulse
profile when I start.

I attempt to combine the above two tools. I make a refinement script
based off of the parameters of EZPrefine, but use PINT as an implementaiton
tool. There will probably be other benefits as well, such as the ability to
supply a .par file instead of having to enter everything by hand.
