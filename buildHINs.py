import networkx as nx

def buildDBLP():
    G = nx.Graph()
    with open("datasets/DBLP/paper_author.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("p"+line[0], type="paper")
            G.add_node("a"+line[1], type="author")
            G.add_edge("p"+line[0], "a"+line[1])
    with open("datasets/DBLP/paper_conference.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("p"+line[0], type="paper")
            G.add_node("c"+line[1], type="conference")
            G.add_edge("p"+line[0], "c"+line[1])
    with open("datasets/DBLP/author_label.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("a"+line[0], type="author")
            G.add_node("l"+line[1], type="label")
            G.add_edge("a"+line[0], "l"+line[1])
    with open("datasets/DBLP/paper_type.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("p"+line[0], type="paper")
            G.add_node("t"+line[1], type="topic")
            G.add_edge("p"+line[0], "t"+line[1])
    return G

def buildDoubanMovie():
    G = nx.Graph()
    with open("datasets/Douban Movie/user_movie.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("u"+line[0], type="user")
            G.add_node("m"+line[1], type="movie")
            G.add_edge("u"+line[0], "m"+line[1])
    with open("datasets/Douban Movie/user_group.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("u"+line[0], type="user")
            G.add_node("g"+line[1], type="group")
            G.add_edge("u"+line[0], "g"+line[1])
    with open("datasets/Douban Movie/movie_actor.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("m"+line[0], type="movie")
            G.add_node("a"+line[1], type="actor")
            G.add_edge("m"+line[0], "a"+line[1])
    with open("datasets/Douban Movie/movie_director.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("m"+line[0], type="movie")
            G.add_node("d"+line[1], type="director")
            G.add_edge("m"+line[0], "d"+line[1])
    with open("datasets/Douban Movie/movie_type.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("m"+line[0], type="movie")
            G.add_node("t"+line[1], type="type")
            G.add_edge("m"+line[0], "t"+line[1])
    with open("datasets/Douban Movie/user_user.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("u"+line[0], type="user")
            G.add_node("u"+line[1], type="user")
            G.add_edge("u"+line[0], "u"+line[1])
    return G

def buildAmazon():
    G = nx.Graph()
    with open("datasets/Amazon/user_item.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("u"+line[0], type="user")
            G.add_node("i"+line[1], type="item")
            G.add_edge("u"+line[0], "i"+line[1])
    with open("datasets/Amazon/item_view.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split(",")[:2]
            G.add_node("i"+line[0], type="item")
            G.add_node("v"+line[1], type="view")
            G.add_edge("i"+line[0], "v"+line[1])
    with open("datasets/Amazon/item_category.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split(",")[:2]
            G.add_node("i"+line[0], type="item")
            G.add_node("c"+line[1], type="category")
            G.add_edge("i"+line[0], "c"+line[1])
    with open("datasets/Amazon/item_brand.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split(",")[:2]
            G.add_node("i"+line[0], type="item")
            G.add_node("b"+line[1], type="brand")
            G.add_edge("i"+line[0], "b"+line[1])
    return G


def buildFreebase():
    G = nx.Graph()
    # linktype2node={0:{"start":"book","end":"book"},1:{"start":"book","end":"film"},2:{"start":"book","end":"sports"},3:{"start":"book","end":"location"},4:{"start":"book","end":"organization"},5:{"start":"film","end":"film"},6:{"start":"music","end":"book"},7:{"start":"music","end":"film"},8:{"start":"music","end":"music"},9:{"start":"music","end":"sports"},10:{"start":"music","end":"location"},11:{"start":"sports","end":"film"},12:{"start":"sports","end":"sports"},13:{"start":"sports","end":"location"},14:{"start":"people","end":"book"},15:{"start":"people","end":"film"},16:{"start":"people","end":"music"},17:{"start":"people","end":"sports"},18:{"start":"people","end":"people"},19:{"start":"people","end":"location"},20:{"start":"people","end":"organization"},21:{"start":"people","end":"business"},22:{"start":"location","end":"film"},23:{"start":"location","end":"location"},24:{"start":"organization","end":"film"},25:{"start":"organization","end":"music"},26:{"start":"organization","end":"sports"},27:{"start":"organization","end":"location"},28:{"start":"organization","end":"organization"},29:{"start":"organization","end":"business"},30:{"start":"business","end":"book"},31:{"start":"business","end":"film"},32:{"start":"business","end":"music"},33:{"start":"business","end":"sports"},34:{"start":"business","end":"location"},35:{"start":"business","end":"business"}}
    # with open("datasets/Freebase/link4.dat", "r") as f:
    #     for line in f.readlines():
    #         line = line.strip().split("\t")[:3]
    #         type = int(line[2])
    #         G.add_node(linktype2node[type]["start"][0]+line[0], type=linktype2node[type]["start"])
    #         G.add_node(linktype2node[type]["end"][0]+line[1], type=linktype2node[type]["end"])
    #         G.add_edge(linktype2node[type]["start"][0]+line[0], linktype2node[type]["end"][0]+line[1])
    # a = set()
    with open("datasets/Freebase/link3.dat", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            starttype = line[2]
            endtype = line[3]
            if starttype == "book":
                line[0] = line[0].replace("b", "bo")
            elif starttype == "business":
                line[0] = line[0].replace("b", "bu")
            if endtype == "book":
                line[1] = line[1].replace("b", "bo")
            elif endtype == "business":
                line[1] = line[1].replace("b", "bu")
            G.add_node(starttype+line[0], type=starttype)
            G.add_node(endtype+line[1], type=endtype)
            G.add_edge(starttype+line[0], endtype+line[1])
    return G

def buildAminer():
    G = nx.Graph()
    with open("datasets/Aminer/paper_author.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("p"+line[0], type="paper")
            G.add_node("a"+line[1], type="author")
            G.add_edge("p"+line[0], "a"+line[1])
    with open("datasets/Aminer/paper_conference.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("p"+line[0], type="paper")
            G.add_node("c"+line[1], type="conference")
            G.add_edge("p"+line[0], "c"+line[1])
    with open("datasets/Aminer/paper_type.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("p"+line[0], type="paper")
            G.add_node("t"+line[1], type="type")
            G.add_edge("p"+line[0], "t"+line[1])
    with open("datasets/Aminer/paper_label.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("p"+line[0], type="paper")
            G.add_node("l"+line[1], type="label")
            G.add_edge("p"+line[0], "l"+line[1])
    with open("datasets/Aminer/paper_year.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")[:2]
            G.add_node("p"+line[0], type="paper")
            G.add_node("y"+line[1], type="year")
            G.add_edge("p"+line[0], "y"+line[1])
    return G

if __name__ == '__main__':
    G = buildAmazon()
    print(G)