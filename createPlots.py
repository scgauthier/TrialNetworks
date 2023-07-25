# from PlottingFcns import plot_TR_from_txt
# from PlottingFcns import plot_multiple_from_text
# from PlottingFcns import plot_max_min_diff
from PlottingFcns import plot_convergence_study

# plot_TR_from_txt('FromRemote/20230209-110841')  # fig 3
# plot_TR_from_txt('FromRemote/20230630-140239')

# plot_multiple_from_text('Fromemote/20230210-112913',
#                         'FromRemote/20230210-112940')

# plot_max_min_diff('FromRemote/20230216-104427',
#                   'FromRemote/20230630-140239',
#                   'FromRemote/20230216-105055',
#                   'FromRemote/20230630-104605',
#                   'FromRemote/20230215-175716',
#                   'FromRemote/20230629-125337')

plot_convergence_study('FromRemote/20230216-104427',
                       'FromRemote/20230216-105055',
                       'FromRemote/20230215-175716')
