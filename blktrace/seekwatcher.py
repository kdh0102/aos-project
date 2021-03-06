#
# Copyright (C) 2007 Oracle.  All rights reserved.
#
# To use seekwatcher, you need to download matplotlib, and have the numpy
# python lib installed on your box (this is the default w/many distro
# matplotlib packages).
#
# There are two basic modes for seekwatcher.  The first is to take
# an existing blktrace file and create a graph.  In this mode the two
# most important options are:
#
# -t (name of the trace file)
# -o (name of the output png)
#
#
# Example:
#
# blktrace -o read_trace -d /dev/sda &
#
# run your test
# kill blktrace
#
# seekwatcher -t read_trace -o trace.png
#
# Seekwatcher can also start blktrace for you, run a command, kill blktrace
# off and generate the plot.  -t and -o are still used, but you also send
# in the program to run and the device to trace.  The trace file is kept,
# so you can plot it again later with different args.
#
# Example:
#
# seekwatcher -t read_trace -o trace.png -p "dd if=/dev/sda of=/dev/zero" \
#       -d /dev/sda
#
# -z allows you to change the window used to zoom in on the most common
# data on the y axis.  Use min:max as numbers in MB where you want to 
# zoom. -z 0:0 forces no zooming at all.  The default tries to find the
# most common area of the disk hit and show only that.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License v2 as published by the Free Software Foundation.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with this program; if not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 021110-1307, USA.
#
import sys, os, signal, time, subprocess, tempfile, signal
from optparse import OptionParser

blktrace_only = False

try:
    from matplotlib import rcParams
    from matplotlib.font_manager import fontManager, FontProperties
    import numpy
except:
    sys.stderr.write("matplotlib not found, using blktrace only mode\n")
    blktrace_only = True

class AnnoteFinder:
  """
  callback for matplotlib to display an annotation when points are clicked on.  The
  point which is closest to the click and within xtol and ytol is identified.
    
  Register this function like this:
    
  scatter(xdata, ydata)
  af = AnnoteFinder(xdata, ydata, annotes)
  connect('button_press_event', af)
  """

  def __init__(self, axis=None):
    if axis is None:
      self.axis = gca()
    else:
      self.axis= axis
    self.drawnAnnotations = {}
    self.links = []
    
  def clear(self):
    for k in self.drawnAnnotations.keys():
        self.drawnAnnotations[k].set_visible(False)

  def __call__(self, event):
    if event.inaxes:
      if event.button != 1:
        self.clear()
        draw()
        return
      clickX = event.xdata
      clickY = event.ydata
      if (self.axis is None) or (self.axis==event.inaxes):
        self.drawAnnote(event.inaxes, clickX, clickY)
    
  def drawAnnote(self, axis, x, y):
    """
    Draw the annotation on the plot
    """
    if self.drawnAnnotations.has_key((x,y)):
      markers = self.drawnAnnotations[(x,y)]
      markers.set_visible(not markers.get_visible())
      draw()
    else:
      t = axis.text(x,y, "(%3.2f, %3.2f)"%(x,y), bbox=dict(facecolor='red',
                    alpha=0.8))
      self.drawnAnnotations[(x,y)] = t
      draw()

def dev2num(dev):
    s2 = dev.replace(',', '.')
    return float(s2)

def flag2num(flag):
    if flag == 'Q':
        return 0.0
    if flag == 'C':
        return 1.0
    if flag == 'U':
        return 2.0
    return 3.0
    sys.stderr.write("unknown flag %s\n" %flag)

def command2num(com):
    if com[0] == 'R':
        return 0.0
    if com[0] == 'W':
        return 1.0
    return 2.0
    sys.stderr.write("unknown command %s\n" % com)

def loaddata(fh,delimiter=None, converters=None):

    def iter(fh, delimiter, converters):
        global devices_sector_max

        if converters is None: converters = {}
        last_sector = None
        last_rw = None
        last_row = None
        last_end = None
        last_cmd = None
        last_size = None
        last_dev = None
        for i,line in enumerate(fh):
            if not line.startswith('C'):
                continue
            row = [converters.get(i,float)(val) for i,val in enumerate(line.split(delimiter))]
            this_time = row[7]
            this_dev = row[8]
            this_sector = row[4]
            this_rw = row[1]
            this_size = row[5] / 512

            devices_sector_max[this_dev] = max(this_sector + this_size,
                                    devices_sector_max.get(this_dev, 0));

            if (last_row and this_rw == last_rw and
                this_dev == last_dev and
                this_time - last_time < .5 and last_size < 512 and
                this_sector == last_end):
                last_end += this_size
                last_size += this_size
                last_row[5] += row[5]
                continue
                
            if last_row:
                for x in last_row:
                    yield x
                
            last_row = row
            last_sector = this_sector
            last_time = this_time
            last_rw = this_rw
            last_end = this_sector + this_size
            last_size = this_size
            last_dev = this_dev
        if last_row:
            for x in last_row:
                yield x

    X = numpy.fromiter(iter(fh, delimiter, converters), dtype=float)
    return X

def sort_by_time(data):
    def sort_iter(sorted):
        for x in sorted:
            for field in data[x]:
                yield field

    times = data[:,7]
    sorted = times.argsort()
    X = numpy.fromiter(sort_iter(sorted), dtype=float)
    shapeit(X)
    return X

def data_movie(data):
    def xycalc(sector):
        if sector < yzoommin or sector > yzoommax:
            return None
        sector = sector - yzoommin
        sector = sector / sectors_per_cell
        yval = floor(sector / num_cells)
        xval = sector % num_cells
        return (xval, yval)

    def add_frame(prev, ins, max):
        if len(prev) > max:
            del prev[0]
        prev.append(ins)

    def graphit(a, prev):
        def plotone(a, x, y, color):
            a.plot(x, y, 's', color=color, mfc=color,
                   mec=color, markersize=options.movie_cell_size)
            a.hold(True)
        alpha = 0.1
        a.hold(False)

        for x in range(len(prev)):
            readx, ready, writex, writey = prev[x]
            if x == len(prev) - 1:
                alpha = 1.0

            if readx:
                color = bluemap(alpha)
                plotone(a, readx, ready, color)
            if writex:
                color = greenmap(alpha)
                plotone(a, writex, writey, color)
            alpha += 0.1
    
    options.movie_cell_size = float(options.movie_cell_size)
    num_cells = 600 / options.movie_cell_size

    total_cells = num_cells * num_cells
    sector_range = yzoommax - yzoommin
    sectors_per_cell = sector_range / total_cells
    total_secs = xmax - xmin
    movie_length = int(options.movie_length)
    movie_fps = int(options.movie_frames)
    total_frames = movie_length * movie_fps
    secs_per_frame = total_secs / total_frames
    print(f"total frames is {total_frames} secs per frame = {secs_per_frame}")
    start_second = xmin

    times = data[:,7]
    figindex = 0

    png_dir = tempfile.mkdtemp(dir=os.path.dirname(options.output))
    movie_name = options.output
    fname, fname_ext = os.path.splitext(options.output)
    fname = os.path.join(png_dir, fname);

    i = 0
    prev = []
    f = figure(figsize=(8,6))
    a = axes([ 0.10, 0.29, .85, .68 ])
    tput_ax = axes([ 0.10, 0.19, .85, .09 ])
    seek_ax = axes([ 0.10, 0.07, .85, .09 ])

    plot_seek_count(seek_ax, None, data, '-', None)
    ticks = seek_ax.get_yticks()
    ticks = list(arange(0, ticks[-1] + ticks[-1]/3, ticks[-1]/3))
    seek_ax.set_yticks(ticks)
    seek_ax.set_yticklabels( [ str(int(x)) for x in ticks ], fontsize='x-small')
    seek_ax.set_ylabel('Seeks / sec', fontsize='x-small')
    seek_ax.set_xlabel('Time (seconds)', fontsize='x-small')
    seek_ax.grid(True)

    plot_throughput(tput_ax, None, data, '-', None)

    # cut down the number of yticks to something more reasonable
    ticks = tput_ax.get_yticks()
    ticks = list(arange(0, ticks[-1] + ticks[-1]/3, ticks[-1]/3))
    tput_ax.set_yticks(ticks)
    tput_ax.set_xticks([])
    tput_ax.grid(True)

    if ticks[-1] < 3:
        tput_ax.set_yticklabels( [ "%.1f" % x for x in ticks ],
                                fontsize='x-small')
    else:
        tput_ax.set_yticklabels( [ "%d" % x for x in ticks ],
                                fontsize='x-small')

    tput_ax.set_ylabel('MB/s', fontsize='x-small')

    a.set_xticklabels([])
    a.set_yticklabels([])
    a.set_xlim(0, num_cells)
    a.set_ylim(0, num_cells)
    a.hold(False)
    datai = 0
    datalen = len(data)
    bluemap = get_cmap("Blues")
    greenmap = get_cmap("Greens")

    while i < total_frames and datai < datalen:
        start = start_second + i * secs_per_frame
        i += 1
        end = start + secs_per_frame
        if datai >= datalen or data[datai][7] > xmax:
            break
        write_xvals = []
        write_yvals = []
        read_xvals = []
        read_yvals = []
        while datai < datalen and data[datai][7] < end:
            row = data[datai]
            time = row[7]
            if time < start:
                print(f"dropping time {time} < start {start}")
                continue
            datai += 1
            sector = row[4]
            size = int(max(row[5] / 512, 1))
            rbs = row[1]
            cell = 0
            while cell < size:
                xy = xycalc(sector)
                sector += sectors_per_cell
                cell += sectors_per_cell
                if xy:
                    if rbs:
                        write_xvals.append(xy[0])
                        write_yvals.append(xy[1])
                    else:
                        read_xvals.append(xy[0])
                        read_yvals.append(xy[1])
        if not read_xvals and not write_xvals:
            continue

        add_frame(prev, (read_xvals, read_yvals, write_xvals, write_yvals), 10)
        graphit(a, prev)

        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_xlim(0, num_cells)
        a.set_ylim(0, num_cells)
        line = seek_ax.axvline(x=end, color='k')
        line2 = tput_ax.axvline(x=end, color='k')
        tput_ax.set_xlim(xmin, xmax)
        seek_ax.set_xlim(xmin, xmax)
        print(f"start {start} secs end {end} secs frame {figindex}")
        f.savefig("%s-%.6d.%s" % (fname, figindex, "png"), dpi=options.dpi)
        line.set_linestyle('None')
        line2.set_linestyle('None')
        figindex += 1
    a.hold(True)

    if mencoder_found == "png2theora" and movie_name.endswith('.ogg'):
        os.system("png2theora -o %s %s" % (movie_name, fname) + '-%06d.png')
    else:
        os.system("mencoder mf://%s*.png -mf type=png:fps=%d -of mpeg -ovc lavc -lavcopts vcodec=mpeg2video:vbitrate=%s -oac copy -o %s" % (fname, movie_fps, options.movie_vbitrate, movie_name))

    for root, dirs, files in os.walk(png_dir):
        for name in files:
            os.remove(os.path.join(root, name))
    os.rmdir(png_dir)

def plot_data(ax, rw, data, style, label, alpha=1):
    def reduce_plot():
        reduce = {}
        skipped = 0
        for i in range(len(times)):
            x = floor(times[i] / x_per_cell)
            y = floor(sectors[i] / y_per_cell)
            if x in reduce and y in reduce[x]:
                skipped += 1
                continue
            y += 1
            h = reduce.setdefault(x, {})
            h[y] = 1
            yield x * x_per_cell
            yield y * y_per_cell
    xcells = 325.0 * options.io_graph_cell_multi
    x_per_cell = (xmax - xmin) / xcells
    ycells = 80.0 * options.io_graph_cell_multi
    y_per_cell = (yzoommax - yzoommin) / ycells

    if rw is None:
        if options.reads_only:
            rw = 0
        if options.writes_only:
            rw = 1
    if rw != None:
        if options.reads_only and rw != 0:
            return
        if options.writes_only and rw != 1:
            return
        rbs = data[:,1]
        data = data[numpy.where(rbs == rw)]
    times = data[:,7]
    sectors = data[:,4]
    if len(times) > 0:
        t = numpy.fromiter(reduce_plot(), dtype=float)
        t.shape = (len(t)//2, 2)
        xdata = t[:,0]
        ydata = t[:,1]
        lines = ax.plot(t[:,0], t[:,1], options.io_graph_dots, mew=0,
                        ms=options.io_graph_marker_size,
                        label=label, alpha=alpha)
        return lines
    return []

def add_roll(roll, max, num):
    if len(roll) == max:
        del roll[0]
    roll.append(num)
    total = 0.0
    for x in roll:
        total += x
    return total / len(roll)

def plot_throughput(ax, rw, data, style, label, alpha=1):
    def tput_iter(sizes,times):
        bytes = 0.0
        sec = None
        roll = []
        for x in range(len(sizes)):
            size = sizes[x]
            cur_time = floor(times[x])
            if sec == None:
                avg = add_roll(roll, options.rolling_avg, 0.0)
                yield (0.0, avg)
                sec = cur_time
                continue
            if sec != cur_time:
                avg = add_roll(roll, options.rolling_avg, bytes)
                yield (sec, avg / (1024 * 1024))
                bytes = 0
                sec = cur_time
            bytes += size
        scale = times[-1] - sec
        if scale > 0 and scale < 1:
            bytes += sizes[-1]
            bytes = bytes / scale
            avg = add_roll(roll, options.rolling_avg, bytes)
            yield(ceil(times[-1]), avg / (1024 * 1024))

    if rw is None:
        if options.reads_only:
            rw = 0
        if options.writes_only:
            rw = 1
    if rw != None:
        if options.reads_only and rw != 0:
            return
        if options.writes_only and rw != 1:
            return
        rbs = data[:,1]
        data = data[numpy.where(rbs == rw)]

    if len(data) == 0:
        return

    times = numpy.array([])
    tput = numpy.array([])
    for x,y in tput_iter(data[:,5], data[:,7]):
        times = numpy.append(times, x)
        tput = numpy.append(tput, y)

    return ax.plot(times, tput, style, label=label, alpha=alpha)

def plot_seek_count(ax, rw, data, style, label, alpha=1):
    def iter(sectors, times):
        count = 0.0
        last_dev = {}
        # last holds an array (sector, size)

        last = None
        last_size = None
        sec = None

        roll = []
        for x in range(len(sectors)):
            sector = sectors[x]
            io_size = data[x][5] / 512
            dev = data[x][8]
            last, last_size = last_dev.get(dev, (None, None))
            cur_time = floor(times[x])
            if sec == None:
                avg = add_roll(roll, options.rolling_avg, 0.0)
                yield (0.0, avg)
                sec = cur_time
                continue
            if sec != cur_time:
                avg = add_roll(roll, options.rolling_avg, count)
                yield (sec, avg)
                count = 0
                sec = cur_time
            if last != None:
                diff = abs((last + last_size) - sector)
                if diff > 128:
                    count += 1
            last_dev[dev] = (sector, io_size)

        scale = times[-1] - sec
        if scale > 0 and scale < 1:
            dev = data[-1][8]
            last, last_size = last_dev[dev]
            sector = sectors[-1]
            diff = abs((last + last_size) - sector)
            if diff > 128:
                count += 1
            count = count / scale
            avg = add_roll(roll, options.rolling_avg, count)
            yield(ceil(times[-1]), avg)

    if rw is None:
        if options.reads_only:
            rw = 0
        if options.writes_only:
            rw = 1

    if rw != None:
        if options.reads_only and rw != 0:
            return
        if options.writes_only and rw != 1:
            return
        rbs = data[:,1]
        data = data[numpy.where(rbs == rw)]

    if len(data) == 0:
        return

    times = numpy.array([])
    counts = numpy.array([])
    for x,y in iter(data[:,4], data[:,7]):
        times = numpy.append(times, x)
        counts = numpy.append(counts, y)

    return ax.plot(times, counts, style, label=label, alpha=alpha)

def run_one_blktrace(trace, device):
    args = [ "blktrace", "-d", device,  "-o", trace, "-b", "2048" ]
    if not options.full_trace:
        args += [ "-a", "complete" ]
    print(" ".join(args))
    return os.spawnlp(os.P_NOWAIT, *args)

def run_blktrace(trace, devices):
    pids = []
    for x in devices:
        tmp = x.replace('/', '.')
        if len(devices) > 1:
            this_trace = trace + "." + tmp
        else:
            this_trace = trace
        pids.append(run_one_blktrace(this_trace, x))
    return pids

blktrace_pids = []
def run_prog(program, trace, devices):
    global blktrace_pids
    def killblktracers(signum, frame):
        global blktrace_pids
        cpy = blktrace_pids
        blktrace_pids = []
        for x in cpy:
            os.kill(x, signal.SIGTERM)
            pid, err = os.wait()
            if err:
                sys.stderr.write("exit due to blktrace failure %d\n" % err)
                sys.exit(1)

    blktrace_pids = run_blktrace(trace, devices)

    # force some IO, blktrace does timestamps from the first IO
    if len(devices) > 1:
        for x in devices:
            try:
                os.system("dd if=%s of=/dev/zero bs=16k count=1 iflag=direct > /dev/null 2>&1" % x)
            except:
                print(f"O_DIRECT read from {x} failed trying buffered")
                b = file(x).read(1024 * 1024)

    signal.signal(signal.SIGTERM, killblktracers)
    signal.signal(signal.SIGINT, killblktracers)
    sys.stderr.write("running :%s:\n" % program)
    os.system(program)
    sys.stderr.write("done running %s\n" % program)
    killblktracers(None, None)
    sys.stderr.write("blktrace done\n")

    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

def run_blkparse(trace, converters):
    tracefiles = []
    data = numpy.array([])
    seen = {}
    print(f"run_blkparse on %s{trace}")
    if not os.path.exists(trace + "blktrace.0"):
        dirname = os.path.dirname(trace) or "."
        files = os.listdir(dirname)
        joinname = os.path.dirname(trace) or ""
        for x in files:
            x = os.path.join(joinname, x)
            if x.startswith(trace) and ".blktrace." in x:
                i = x.rindex('.blktrace.')
                cur = x[0:i]
                if cur not in seen:
                    tracefiles.append(x[0:i])
                    seen[cur] = 1
    else:
        tracefiles.append(trace)

    for x in tracefiles:
        print(f"using tracefile {x}")
        p = os.popen('blkparse -q -i ' + x +
                     ' -f "%a %d %M %m %S %N %s %5T.%9t %D\n"')
        cur = loaddata(p, converters=converters)
        data = numpy.append(data, cur)
    return data

def shapeit(X):
    lines = len(X) // 9
    X.shape = (lines, 9)

def unshapeit(X):
    lines = len(X) * 9
    X.shape = (lines, 1)

def getlabel(i):
    if i < len(options.label):
        return options.label[i]
    return ""

def line_picker(line, mouseevent):
    if mouseevent.xdata is None: return False, dict()
    print(f"{mouseevent.xdata} {mouseevent.ydata}")
    return False, dict()

def running_config():
	"""
	Return path of config file of the currently running kernel
	"""
	version = subprocess.getoutput('uname -r')
	for config in ('/proc/config.gz', \
                       '/boot/config-%s' % version,
                       '/lib/modules/%s/build/.config' % version):
		if os.path.isfile(config):
			return config
	return None


def check_for_kernel_feature(feature):
	config = running_config()

	if not config:
		sys.stderr.write("Can't find kernel config file")

	if config.endswith('.gz'):
		grep = 'zgrep'
	else:
		grep = 'grep'
	grep += ' ^CONFIG_%s= %s' % (feature, config)

	if not subprocess.getoutput(grep):
		sys.stderr.write("Kernel doesn't have a %s feature\n" % (feature))
		sys.exit(1)

def check_for_debugfs():
    tmp = subprocess.getoutput('mount | grep /sys/kernel/debug')
    tmp = len(tmp)
    if tmp == 0:
        sys.stderr.write("debugfs not mounted (/sys/kernel/debug)\n")
        sys.exit(1)

def check_for_mencoder():
    dirs = os.getenv('PATH', os.path.defpath).split(os.path.pathsep)
    for dir in dirs:
        fname = os.path.join(dir, 'png2theora')
        if os.path.isfile(fname):
            return "png2theora"
    for dir in dirs:
        fname = os.path.join(dir, 'mencoder')
        if os.path.isfile(fname):
            return "mencoder"
    return None

def translate_sector(dev, sector):
    return device_translate[dev] + sector;

usage = "usage: %prog [options]"
parser = OptionParser(usage=usage)
parser.add_option("-d", "--device", help="Device for blktrace", default=[],
                  action="append")
parser.add_option("-t", "--trace", help="blktrace file", default=[],
                  action="append")
parser.add_option("-p", "--prog", help="exec program", default="")
parser.add_option("", "--full-trace", help="Don't filter blktrace events",
                  default=False, action="store_true")

if not blktrace_only:
    parser.add_option("-z", "--zoom", help="Zoom range min:max (in MB)",
                      default="")
    parser.add_option("-x", "--xzoom", help="Time range min:max (seconds)",
                    default="")
    parser.add_option("-o", "--output", help="output file", default="trace.png")
    parser.add_option("-l", "--label", help="label", default=[],
                      action="append")
    parser.add_option("", "--dpi", help="dpi", default=120)
    parser.add_option("", "--io-graph-dots", help="Disk IO dot style",
                      default='s')
    parser.add_option("", "--io-graph-marker-size", help="Disk IO dot size",
                      default=1.5, type="float")
    parser.add_option("", "--io-graph-cell-multi", help="Multiplier for cells",
                      default=2, type="float")
    parser.add_option("-I", "--no-io-graph", help="Don't create an IO graph",
                      default=False, action="store_true")
    parser.add_option("-r", "--rolling-avg",
                  help="Rolling average for seeks and throughput (in seconds)",
                  default=None)

    parser.add_option("-i", "--interactive", help="Use matplotlib interactive",
                      action="store_true", default=False)
    parser.add_option("", "--backend",
               help="matplotlib backend (QtAgg, TkAgg, GTKAgg) case sensitive",
               default="QtAgg")
    parser.add_option("-T", "--title", help="Graph Title", default="")
    parser.add_option("-R", "--reads-only", help="Graph only reads",
                      default=False, action="store_true")
    parser.add_option("-W", "--writes-only", help="Graph only writes",
                      default=False, action="store_true")

    mencoder_found = check_for_mencoder()
    if mencoder_found:
        parser.add_option("-m", "--movie", help="Generate an IO movie",
                          default=False, action="store_true")
        parser.add_option("", "--movie-frames",
                          help="Number of frames per second",
                          default=10)
        parser.add_option("", "--movie-length", help="Movie length in seconds",
                          default=30)
        parser.add_option("", "--movie-cell-size",
                          help="Size in pixels of the IO cells", default=2)
        parser.add_option("", "--movie-vbitrate",
                          help="Mencoder vbitrate option (default 16000)",
                          default="16000")

(options,args) = parser.parse_args()

if not blktrace_only:
    # rcParams['numerix'] = 'numpy'
    if options.interactive:
        rcParams['backend'] = options.backend
        rcParams['interactive'] = 'True'
    else:
        rcParams['backend'] = 'Agg'
        rcParams['interactive'] = 'False'
    from pylab import *

if not options.trace:
    parser.print_help()
    sys.exit(1)

converters = {}
converters[0] = flag2num
converters[1] = command2num
converters[8] = dev2num

if options.prog:
    check_for_kernel_feature("DEBUG_FS")
    check_for_kernel_feature("BLK_DEV_IO_TRACE")
    check_for_debugfs()

    if not options.trace or not options.device:
        sys.stderr.write("blktrace output file or device not specified\n")
        sys.exit(1)
    run_prog(options.prog, options.trace[0], options.device)
    if blktrace_only:
        sys.exit(0)

    if not options.title:
        options.title = options.prog

data = numpy.array([])
runs = []
must_sort = True

for x in options.trace:
    devices_sector_max = {}
    run = run_blkparse(x, converters)

    device_translate = {}
    total = 0
    if len(devices_sector_max) > 1:
        must_sort = True
        for x in devices_sector_max:
            device_translate[x] = total + devices_sector_max[x]
            total += devices_sector_max[x]
    shapeit(run)
    if len(devices_sector_max) > 1:
        for x in run:
            sector = x[4]
            dev = x[8]
            x[4] = device_translate[dev] + sector
        
    sorted = sort_by_time(run)
    run = sorted

    unshapeit(run)
    runs.append(run)
    data = numpy.append(data, run)

shapeit(data)
for x in runs:
    shapeit(x)
    if len(x) == 0:
        sys.stderr.write("Empty blktrace run found, exiting\n")
        sys.exit(1)

if must_sort:
    sorted = sort_by_time(data)
    data = sorted

# try to drop out the least common data points by creating
# a historgram of the sectors seen.
sectors = data[:,4]
sizes = data[:,5]
ymean = numpy.mean(sectors)
sectormax = numpy.max(sectors)
sectormin = numpy.min(sectors)

if not options.zoom or ':' not in options.zoom:
    def add_range(hist, step, sectormin, start, size):
        while size > 0:
            slot = int((start - sectormin) / step)
            slot_start = step * slot + sectormin
            if slot >= len(hist) or slot < 0:
                sys.stderr.write("illegal slot %d start %d step %d\n" %
                                (slot, start, step))
                return
            else:
                val = hist[slot]
            this_size = min(size, start - slot_start)
            this_count = max(this_size / 512, 1)
            hist[slot] = val + this_count
            size -= this_size
            start += this_count
        
    hist = [0] * 11
    step = (sectormax - sectormin) / 10
    for row in data:
        start = row[4]
        size = row[5] / 512
        add_range(hist, step, sectormin, start, size)

    m = max(hist)

    for x in range(len(hist)):
        if m == hist[x]:
            maxi = x
    # hist[maxi] is the most common bucket.  walk toward it from the
    # min and max values looking for the first buckets that have some
    # significant portion of the data
    #
    yzoommin = maxi * step + sectormin
    for x in range(0, maxi):
        if hist[x] > hist[maxi] * .05:
            yzoommin = x * step + sectormin
            break

    yzoommax = (maxi + 1) * step + sectormin
    for x in range(len(hist) - 1, maxi, -1):
        if hist[x] > hist[maxi] * .05:
            yzoommax = (x + 1) * step + sectormin
            break
else:
    words = options.zoom.split(':')
    yzoommin = max(0, float(words[0]) * 2048)
    if float(words[1]) == 0:
        yzoommax = sectormax
    else:
        yzoommax = min(sectormax, float(words[1]) * 2048)

sizes = 0
flags = [ x[:,0] for x in runs ]
times = data[:,7]
xmin = numpy.min(times)
xmax = numpy.max(times)

if options.rolling_avg == None:
    options.rolling_avg = max(1, int((xmax - xmin) / 25))
else:
    options.rolling_avg = max(1, int(options.rolling_avg))

if options.xzoom:
    words = [ float(x) for x in options.xzoom.split(':') ]
    if words[0] != 0:
        xmin = words[0]
    if words[1] != 0:
        xmax = words[1]

sectors = 0
flags = 0
completed = 0
times = 0

if options.no_io_graph:
    total_graphs = 2
else:
    total_graphs = 3

if mencoder_found and options.movie:
    data_movie(runs[0])
    sys.exit(1)

f = figure(figsize=(8,6))

if options.title:
    options.title += "\n\n"

# Throughput goes at the botoom
a = subplot(total_graphs, 1, total_graphs)
for i in range(len(runs)):
    label = getlabel(i)
    plot_throughput(a, None, runs[i], '-', label)

# make sure the final second goes on the x axes
ticks = list(arange(xmin, xmax, xmax/8))
ticks.append(xmax)
xticks = ticks
a.set_xticks(ticks)
a.set_yticklabels( [ "%d" % x for x in ticks ])
if ticks[-1] < 4:
    xticklabels = [ "%.1f" % x for x in ticks ]
else:
    xticklabels = [ "%d" % x for x in ticks ]
a.set_xticklabels(xticklabels)

# cut down the number of yticks to something more reasonable
ticks = a.get_yticks()
ticks = list(arange(0, ticks[-1] + ticks[-1]/4, ticks[-1]/4))
a.set_yticks(ticks)

if ticks[-1] < 4:
    a.set_yticklabels( [ "%.1f" % x for x in ticks ])
else:
    a.set_yticklabels( [ "%d" % x for x in ticks ])

a.set_title('Throughput')
a.set_ylabel('MB/s')

# the bottom graph gets xticks, set it here
a.set_xlabel('Time (seconds)')
if options.label:
    a.legend(loc=(1.01, 0.5), shadow=True, pad=0.5, numpoints=2,
                  handletextsep = 0.005,
                  labelsep = 0.01,
                  prop=FontProperties(size='x-small') )

# next is the seek count graph
a = subplot(total_graphs, 1, total_graphs - 1)
for i in range(len(runs)):
    label = getlabel(i)
    plot_seek_count(a, None, runs[i], '-', label)

# cut down the number of yticks to something more reasonable
ticks = a.get_yticks()
ticks = list(arange(0, ticks[-1] + ticks[-1]/4, ticks[-1]/4))
a.set_yticks(ticks)
a.set_yticklabels( [ str(int(x)) for x in ticks ])

if options.no_io_graph and options.title:
    a.set_title(options.title + 'Seek Count')
else:
    a.set_title('Seek Count')

a.set_ylabel('Seeks / sec')
if options.label:
    a.legend(loc=(1.01, 0.5), shadow=True, pad=0.5, numpoints=2,
                  handletextsep = 0.005,
                  labelsep = 0.01,
                  prop=FontProperties(size='x-small') )

# and the optional IO graph
if not options.no_io_graph:
    a = subplot(total_graphs, 1, total_graphs - 2)
    for i in range(len(runs)):
        label = getlabel(i)
        plot_data(a, 0, runs[i], options.io_graph_dots, label + " Read")
        plot_data(a, 1, runs[i], options.io_graph_dots, label + " Write")

    af = AnnoteFinder(axis=a)
    connect('button_press_event', af)
    a.set_title(options.title + 'Disk IO')
    a.set_ylabel('Disk offset (MB)')
    flag = data[:,0]
    sectors = data[:,4]
    zoom = (sectors > yzoommin) & (sectors < yzoommax)
    zoom = data[zoom]
    sectors = zoom[:,4]
    yzoommin = numpy.min(sectors)
    yzommmax = numpy.max(sectors)
    ticks = list(arange(yzoommin, yzoommax, (yzoommax - yzoommin) / 4))
    ticks.append(yzoommax)
    a.set_yticks(ticks)
    a.set_yticklabels( [ str(int(x/2048)) for x in ticks ] )
    a.legend(loc=(1.01, 0.5), shadow=True, numpoints=1,
                  markerscale = 1.1,
                  prop=FontProperties(size='x-small') )
    a.set_ylim(yzoommin, yzoommax)

# squeeze the graphs over to the left a bit to make room for the
# legends
#
subplots_adjust(right = 0.8, hspace=0.3)

# finally, some global bits for each subplot
for x in range(1, total_graphs + 1):
    a = subplot(total_graphs, 1, x)

    # turn off the xtick labels on the graphs above the bottom
    if not options.interactive and x < total_graphs:
        a.set_xticklabels([])
    elif options.interactive:
        a.set_xticks(xticks)
        a.set_xticklabels(xticklabels)

    # create dashed lines for each ytick
    ticks = a.get_yticks()
    ymin, ymax = a.get_ylim()
    for y in ticks[1:]:
        try:
            a.hlines(y, xmin, xmax, linestyle='dashed', alpha=0.5)
        except:
            a.hlines(y, xmin, xmax, alpha=0.5)
    a.set_ylim(ymin, ymax)
    # set the xlimits to something sane
    a.set_xlim(xmin, xmax)

if not options.interactive:
    print(f"saving graph to %s{options.output}")
    savefig(options.output, dpi=options.dpi, orientation='landscape')
show()
