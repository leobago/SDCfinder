# SDCfinder

This tool aims to detect Silent Data Corruption (SDC) ocurring in
supercomputers and large datacenters. The current implementation is in C but we
will be adding a Python version soon. We also are planning to cover GPU memory
as well. The way this tool works is rather simple. At the begining of the
execution two hexadecimal paterns will be selected randomly. Then it allocates
a large buffer of the available memory. Then, it will iterate doing the three
following actions: write, sleep and read. When reading it will check whether
any of the read patterns is different from the one written, if it is the case
it will log it as a detected SDC. The next iteration the second pattern will be
used for writing and it will continue swtiching between both paterns until the
end of the execution.


